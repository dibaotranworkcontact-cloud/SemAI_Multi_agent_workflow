"""
RAG (Retrieval-Augmented Generation) System for Semai Crew
Provides semantic search and retrieval of documentation, models, and code.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import warnings

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class Document:
    """Represents a document in the RAG system"""
    id: str
    content: str
    title: str
    source: str
    doc_type: str  # 'markdown', 'code', 'api', 'reference'
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_index: int = 0
    total_chunks: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        if d['embedding'] is not None:
            d['embedding'] = d['embedding'].tolist()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        if data.get('embedding'):
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


class DocumentLoader:
    """Loads and chunks documents from various sources"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents = []
    
    def load_markdown_files(self, directory: str) -> List[Document]:
        """Load all markdown files from directory"""
        docs = []
        md_files = list(Path(directory).glob("*.md"))
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                title = file_path.stem.replace('_', ' ')
                chunks = self._chunk_text(content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        id=f"md_{file_path.stem}_{chunk_idx}",
                        content=chunk,
                        title=title,
                        source=str(file_path),
                        doc_type="markdown",
                        metadata={"file": file_path.name},
                        chunk_index=chunk_idx,
                        total_chunks=len(chunks)
                    )
                    docs.append(doc)
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}")
        
        return docs
    
    def load_python_files(self, directory: str) -> List[Document]:
        """Load Python files and extract docstrings and comments"""
        docs = []
        py_files = list(Path(directory).glob("**/*.py"))
        
        for file_path in py_files:
            if '__pycache__' in str(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract docstrings and comments
                extracted = self._extract_python_knowledge(content)
                title = f"Code: {file_path.stem}"
                
                chunks = self._chunk_text(extracted)
                
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        id=f"py_{file_path.stem}_{chunk_idx}",
                        content=chunk,
                        title=title,
                        source=str(file_path),
                        doc_type="code",
                        metadata={"file": file_path.name, "type": "python"},
                        chunk_index=chunk_idx,
                        total_chunks=len(chunks)
                    )
                    docs.append(doc)
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}")
        
        return docs
    
    def load_model_metadata(self, builtin_models_path: str) -> List[Document]:
        """Extract model information from builtin_models.py"""
        docs = []
        
        try:
            with open(builtin_models_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract class definitions and their docstrings
            classes_info = self._extract_class_info(content)
            
            for class_name, info in classes_info.items():
                doc = Document(
                    id=f"model_{class_name}",
                    content=info,
                    title=f"Model: {class_name}",
                    source=builtin_models_path,
                    doc_type="api",
                    metadata={"class": class_name, "type": "model"}
                )
                docs.append(doc)
        except Exception as e:
            warnings.warn(f"Failed to load model metadata: {e}")
        
        return docs
    
    def load_custom_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Load custom documents"""
        docs = []
        
        for idx, doc_data in enumerate(documents):
            doc = Document(
                id=f"custom_{idx}",
                content=doc_data.get('content', ''),
                title=doc_data.get('title', 'Custom Document'),
                source=doc_data.get('source', 'custom'),
                doc_type=doc_data.get('type', 'reference'),
                metadata=doc_data.get('metadata', {})
            )
            docs.append(doc)
        
        return docs
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            
            # Move start position (overlap)
            start = end - self.overlap
            
            # Avoid infinite loop on small texts
            if end == len(text):
                break
        
        return chunks
    
    def _extract_python_knowledge(self, content: str) -> str:
        """Extract docstrings, comments, and class/function names"""
        lines = content.split('\n')
        extracted = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Include docstrings and comments
            if stripped.startswith('"""') or stripped.startswith("'''"):
                extracted.append(line)
            elif stripped.startswith('#'):
                extracted.append(line)
            elif any(stripped.startswith(kw) for kw in ['class ', 'def ', '@']):
                extracted.append(line)
        
        return '\n'.join(extracted)
    
    def _extract_class_info(self, content: str) -> Dict[str, str]:
        """Extract class definitions and docstrings"""
        import re
        
        classes = {}
        class_pattern = r'class\s+(\w+).*?:\s*\n\s*"""(.*?)"""'
        
        matches = re.finditer(class_pattern, content, re.DOTALL)
        for match in matches:
            class_name = match.group(1)
            docstring = match.group(2)
            classes[class_name] = f"Class: {class_name}\n\nDocstring:\n{docstring}"
        
        return classes


class RAGEmbedder:
    """Manages embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts"""
        return self.model.encode(texts, convert_to_numpy=True)


class RAGRetriever:
    """Retrieves relevant documents using semantic search"""
    
    def __init__(self, documents: List[Document], embedder: Optional[RAGEmbedder] = None, 
                 similarity_threshold: float = 0.3):
        """
        Initialize retriever.
        
        Args:
            documents: List of Document objects
            embedder: RAGEmbedder instance (if None, uses simple text matching)
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.documents = documents
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.use_semantic = embedder is not None and SKLEARN_AVAILABLE
        
        if self.use_semantic:
            self._embed_documents()
    
    def _embed_documents(self):
        """Embed all documents"""
        texts = [doc.content for doc in self.documents]
        embeddings = self.embedder.embed_batch(texts)
        
        for doc, embedding in zip(self.documents, embeddings):
            doc.embedding = embedding
    
    def retrieve(self, query: str, top_k: int = 5, 
                doc_type: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k most relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            doc_type: Filter by document type (markdown, code, api, reference)
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        results = []
        
        if self.use_semantic:
            results = self._semantic_search(query, top_k, doc_type)
        else:
            results = self._text_search(query, top_k, doc_type)
        
        return results
    
    def _semantic_search(self, query: str, top_k: int, 
                        doc_type: Optional[str]) -> List[Tuple[Document, float]]:
        """Semantic search using embeddings"""
        query_embedding = self.embedder.embed(query)
        
        # Filter documents by type if specified
        candidates = self.documents
        if doc_type:
            candidates = [d for d in candidates if d.doc_type == doc_type]
        
        if not candidates:
            return []
        
        # Compute similarities
        embeddings = np.array([doc.embedding for doc in candidates])
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices[:top_k]:
            score = float(similarities[idx])
            if score >= self.similarity_threshold:
                results.append((candidates[idx], score))
        
        return results
    
    def _text_search(self, query: str, top_k: int, 
                    doc_type: Optional[str]) -> List[Tuple[Document, float]]:
        """Keyword-based text search (fallback)"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            if doc_type and doc.doc_type != doc_type:
                continue
            
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            
            if overlap > 0:
                score = overlap / len(query_words)
                if score >= self.similarity_threshold:
                    results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def retrieve_by_source(self, source: str) -> List[Document]:
        """Retrieve all documents from a specific source"""
        return [doc for doc in self.documents if source in doc.source]
    
    def retrieve_by_type(self, doc_type: str) -> List[Document]:
        """Retrieve all documents of a specific type"""
        return [doc for doc in self.documents if doc.doc_type == doc_type]


class RAGSystem:
    """Complete RAG system combining loader, embedder, and retriever"""
    
    def __init__(self, workspace_root: str, use_embeddings: bool = True, 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system.
        
        Args:
            workspace_root: Root directory of the workspace
            use_embeddings: Whether to use semantic embeddings
            embedding_model: HuggingFace model name
        """
        self.workspace_root = Path(workspace_root)
        self.documents = []
        self.embedder = None
        self.retriever = None
        self.cache_path = self.workspace_root / ".rag_cache"
        
        # Initialize embedder if available
        if use_embeddings and EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = RAGEmbedder(embedding_model)
            except Exception as e:
                warnings.warn(f"Failed to initialize embedder: {e}. Using keyword search.")
        
        self.loader = DocumentLoader()
    
    def build_index(self, load_markdown: bool = True, load_code: bool = True, 
                   load_models: bool = True, use_cache: bool = True) -> int:
        """
        Build the RAG index by loading documents.
        
        Args:
            load_markdown: Whether to load markdown documentation
            load_code: Whether to load Python code
            load_models: Whether to load model metadata
            use_cache: Whether to use cached embeddings
            
        Returns:
            Number of documents loaded
        """
        # Try loading from cache
        if use_cache and self.cache_path.exists():
            try:
                return self._load_from_cache()
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}. Rebuilding index...")
        
        # Load documents
        if load_markdown:
            md_docs = self.loader.load_markdown_files(str(self.workspace_root))
            self.documents.extend(md_docs)
        
        if load_code:
            code_docs = self.loader.load_python_files(str(self.workspace_root / "src" / "semai"))
            self.documents.extend(code_docs)
        
        if load_models:
            model_docs = self.loader.load_model_metadata(
                str(self.workspace_root / "src" / "semai" / "builtin_models.py")
            )
            self.documents.extend(model_docs)
        
        # Create retriever
        self.retriever = RAGRetriever(self.documents, self.embedder)
        
        # Cache if using embeddings
        if use_cache and self.embedder:
            self._save_to_cache()
        
        return len(self.documents)
    
    def search(self, query: str, top_k: int = 5, 
              doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results
            doc_type: Filter by type (markdown, code, api, reference)
            
        Returns:
            List of result dictionaries
        """
        if not self.retriever:
            raise RuntimeError("RAG index not built. Call build_index() first.")
        
        results = self.retriever.retrieve(query, top_k, doc_type)
        
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "source": doc.source,
                "type": doc.doc_type,
                "similarity": score,
                "full_content": doc.content
            }
            for doc, score in results
        ]
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get context string for LLM augmentation.
        
        Args:
            query: Search query
            top_k: Number of documents to include
            
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for result in results:
            context_parts.append(
                f"Source: {result['source']}\n"
                f"Title: {result['title']}\n"
                f"Content:\n{result['full_content']}\n"
                f"Relevance: {result['similarity']:.2%}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _save_to_cache(self):
        """Save documents and embeddings to cache"""
        try:
            self.cache_path.mkdir(exist_ok=True)
            
            docs_data = [doc.to_dict() for doc in self.documents]
            
            cache_file = self.cache_path / "documents.json"
            with open(cache_file, 'w') as f:
                json.dump(docs_data, f, indent=2)
            
            metadata_file = self.cache_path / "metadata.json"
            metadata = {
                "count": len(self.documents),
                "embedding_model": self.embedder.model_name if self.embedder else None,
                "timestamp": datetime.now().isoformat()
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")
    
    def _load_from_cache(self) -> int:
        """Load documents from cache"""
        cache_file = self.cache_path / "documents.json"
        
        if not cache_file.exists():
            return 0
        
        with open(cache_file, 'r') as f:
            docs_data = json.load(f)
        
        self.documents = [Document.from_dict(doc_data) for doc_data in docs_data]
        self.retriever = RAGRetriever(self.documents, self.embedder)
        
        return len(self.documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        doc_types = {}
        for doc in self.documents:
            doc_types[doc.doc_type] = doc_types.get(doc.doc_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "by_type": doc_types,
            "has_embeddings": self.embedder is not None,
            "embedding_model": self.embedder.model_name if self.embedder else None,
            "cache_path": str(self.cache_path) if self.cache_path.exists() else None
        }


# Global RAG system instance
_rag_system = None


def initialize_rag(workspace_root: str, use_embeddings: bool = True) -> RAGSystem:
    """Initialize global RAG system"""
    global _rag_system
    _rag_system = RAGSystem(workspace_root, use_embeddings)
    _rag_system.build_index()
    return _rag_system


def get_rag_system() -> RAGSystem:
    """Get global RAG system instance"""
    global _rag_system
    if _rag_system is None:
        raise RuntimeError("RAG system not initialized. Call initialize_rag() first.")
    return _rag_system
