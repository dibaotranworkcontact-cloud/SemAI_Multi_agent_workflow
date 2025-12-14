"""
Corrective Augmented Generation (CAG) Tool for CrewAI

This module implements a comprehensive CAG system that validates and improves
generated content through multi-criteria evaluation and iterative augmentation.

Components:
- CorrectivenessValidator: Multi-criteria content validation
- AugmentationEngine: Content improvement strategies
- CAGTool: CrewAI-compatible orchestrator
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import json
from datetime import datetime
from crewai.tools import BaseTool
import re


class ValidationCriteria(Enum):
    """Enumeration of validation criteria for content assessment."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"


class AugmentationStrategy(Enum):
    """Enumeration of augmentation strategies for content improvement."""
    IMPROVE = "improve"          # Enhance existing content quality
    EXPAND = "expand"            # Add more detail and depth
    SIMPLIFY = "simplify"        # Make more concise and understandable
    CORRECT = "correct"          # Fix errors and inconsistencies
    CLARIFY = "clarify"          # Improve clarity and readability


@dataclass
class ValidationResult:
    """Data structure for content validation results."""
    criterion: ValidationCriteria
    score: float  # 0.0 to 1.0
    feedback: str
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "criterion": self.criterion.value,
            "score": self.score,
            "feedback": self.feedback,
            "issues": self.issues
        }


@dataclass
class AugmentationResult:
    """Data structure for augmentation operation results."""
    strategy: AugmentationStrategy
    original_content: str
    improved_content: str
    quality_improvement: float  # 0.0 to 1.0
    changes_made: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert augmentation result to dictionary."""
        return {
            "strategy": self.strategy.value,
            "original_length": len(self.original_content),
            "improved_length": len(self.improved_content),
            "quality_improvement": self.quality_improvement,
            "changes_made": self.changes_made
        }


@dataclass
class CAGResult:
    """Comprehensive result from CAG processing."""
    original_content: str
    final_content: str
    quality_score: float  # 0.0 to 1.0
    validation_results: Dict[str, ValidationResult]
    augmentation_history: List[AugmentationResult] = field(default_factory=list)
    iterations_used: int = 0
    reference_content: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CAG result to dictionary."""
        return {
            "original_content": self.original_content,
            "final_content": self.final_content,
            "quality_score": self.quality_score,
            "iterations_used": self.iterations_used,
            "validation_results": {k: v.to_dict() for k, v in self.validation_results.items()},
            "augmentation_history": [a.to_dict() for a in self.augmentation_history],
            "reference_content": self.reference_content,
            "timestamp": self.timestamp
        }


class CorrectivenessValidator:
    """
    Validates content against multiple quality criteria.
    
    Provides multi-dimensional assessment of content quality including
    accuracy, completeness, clarity, consistency, and relevance.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the correctness validator.
        
        Args:
            verbose: Enable detailed validation feedback
        """
        self.verbose = verbose
        self.validation_history: List[Tuple[str, Dict]] = []
    
    def validate_accuracy(self, content: str, reference: Optional[str] = None) -> ValidationResult:
        """
        Validate content accuracy.
        
        Args:
            content: Content to validate
            reference: Reference content for comparison (optional)
            
        Returns:
            ValidationResult with accuracy assessment
        """
        issues = []
        
        # Check for logical consistency
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) < 1:
            issues.append("Content too short to assess")
            score = 0.3
        else:
            # Basic accuracy checks
            if any(word in content.lower() for word in ["unclear", "unknown", "cannot determine"]):
                issues.append("Content contains uncertainty markers")
            
            # Check for contradictions
            lines = content.split('\n')
            if len(lines) > 1:
                contradiction_found = False
                for i, line1 in enumerate(lines):
                    for line2 in lines[i+1:]:
                        if "not" in line1.lower() and line1.split("not")[0] in line2.lower():
                            contradiction_found = True
                            issues.append("Possible contradiction detected")
                            break
                
                if reference and len(reference) > 0:
                    similarity = self._calculate_similarity(content, reference)
                    if similarity < 0.3:
                        issues.append("Content deviates significantly from reference")
            
            # Calculate score based on issues
            score = 1.0 - (len(issues) * 0.15)
            score = max(0.0, min(1.0, score))
        
        feedback = f"Accuracy assessment: {len(issues)} potential issues identified"
        result = ValidationResult(
            criterion=ValidationCriteria.ACCURACY,
            score=score,
            feedback=feedback,
            issues=issues
        )
        
        if self.verbose:
            print(f"Accuracy Score: {score:.2f} - {feedback}")
        
        return result
    
    def validate_completeness(self, content: str, reference: Optional[str] = None) -> ValidationResult:
        """
        Validate content completeness.
        
        Args:
            content: Content to validate
            reference: Reference content for comparison (optional)
            
        Returns:
            ValidationResult with completeness assessment
        """
        issues = []
        
        word_count = len(content.split())
        
        # Check minimum content length
        if word_count < 20:
            issues.append("Content too brief (< 20 words)")
            score = 0.4
        elif word_count < 50:
            issues.append("Content may lack sufficient detail (< 50 words)")
            score = 0.6
        else:
            # Check for coverage
            required_elements = ["introduction", "detail", "conclusion"]
            has_intro = any(phrase in content.lower() for phrase in ["first", "begin", "start", "introduce"])
            has_detail = word_count > 100
            has_conclusion = any(phrase in content.lower() for phrase in ["conclude", "summary", "final", "end"])
            
            elements_found = sum([has_intro, has_detail, has_conclusion])
            score = (elements_found / 3.0) * 0.7 + 0.3
            
            if not has_intro:
                issues.append("Missing introductory content")
            if not has_detail:
                issues.append("Insufficient detail and explanation")
            if not has_conclusion:
                issues.append("Missing concluding statement")
        
        feedback = f"Completeness assessment: {word_count} words, {len(issues)} gaps identified"
        result = ValidationResult(
            criterion=ValidationCriteria.COMPLETENESS,
            score=score,
            feedback=feedback,
            issues=issues
        )
        
        if self.verbose:
            print(f"Completeness Score: {score:.2f} - {feedback}")
        
        return result
    
    def validate_clarity(self, content: str) -> ValidationResult:
        """
        Validate content clarity and readability.
        
        Args:
            content: Content to validate
            
        Returns:
            ValidationResult with clarity assessment
        """
        issues = []
        
        # Check sentence complexity
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length > 25:
            issues.append(f"Long sentences (avg {avg_sentence_length:.0f} words)")
        elif avg_sentence_length < 5:
            issues.append(f"Very short sentences (avg {avg_sentence_length:.0f} words)")
        
        # Check for jargon and complexity
        complex_patterns = r'\b\w{15,}\b'  # Words longer than 15 characters
        complex_words = len(re.findall(complex_patterns, content))
        if complex_words > len(words := content.split()) * 0.15:
            issues.append(f"High use of complex vocabulary ({complex_words} long words)")
        
        # Check punctuation balance
        open_parens = content.count('(')
        close_parens = content.count(')')
        if open_parens != close_parens:
            issues.append("Unbalanced parentheses")
        
        score = 1.0 - (len(issues) * 0.2)
        score = max(0.0, min(1.0, score))
        
        feedback = f"Clarity assessment: avg sentence {avg_sentence_length:.1f} words, {len(issues)} clarity issues"
        result = ValidationResult(
            criterion=ValidationCriteria.CLARITY,
            score=score,
            feedback=feedback,
            issues=issues
        )
        
        if self.verbose:
            print(f"Clarity Score: {score:.2f} - {feedback}")
        
        return result
    
    def validate_consistency(self, content: str) -> ValidationResult:
        """
        Validate content internal consistency.
        
        Args:
            content: Content to validate
            
        Returns:
            ValidationResult with consistency assessment
        """
        issues = []
        lines = content.split('\n')
        
        # Check for tense consistency
        past_tense_count = len(re.findall(r'\b(was|were|had|did)\b', content, re.IGNORECASE))
        present_tense_count = len(re.findall(r'\b(is|are|has|do)\b', content, re.IGNORECASE))
        
        if past_tense_count > present_tense_count * 2:
            issues.append("Primarily past tense - check for tense consistency")
        elif present_tense_count > past_tense_count * 2:
            issues.append("Primarily present tense - check for tense consistency")
        
        # Check for terminology consistency
        potential_duplicates = {}
        words = [w.lower() for w in content.split() if len(w) > 5]
        for word in set(words):
            count = words.count(word)
            if count > 5:
                potential_duplicates[word] = count
        
        if len(potential_duplicates) > 0:
            most_repeated = max(potential_duplicates.items(), key=lambda x: x[1])
            if most_repeated[1] > 8:
                issues.append(f"Repetitive word usage: '{most_repeated[0]}' used {most_repeated[1]} times")
        
        score = 1.0 - (len(issues) * 0.25)
        score = max(0.0, min(1.0, score))
        
        feedback = f"Consistency assessment: {len(issues)} consistency issues found"
        result = ValidationResult(
            criterion=ValidationCriteria.CONSISTENCY,
            score=score,
            feedback=feedback,
            issues=issues
        )
        
        if self.verbose:
            print(f"Consistency Score: {score:.2f} - {feedback}")
        
        return result
    
    def validate_relevance(self, content: str, reference: Optional[str] = None) -> ValidationResult:
        """
        Validate content relevance to topic.
        
        Args:
            content: Content to validate
            reference: Reference content for comparison (optional)
            
        Returns:
            ValidationResult with relevance assessment
        """
        issues = []
        
        if not reference:
            # Generic relevance check
            if len(content.split()) < 20:
                issues.append("Content too brief for relevance assessment")
            score = 0.7
        else:
            # Compare with reference
            similarity = self._calculate_similarity(content, reference)
            
            if similarity > 0.8:
                feedback_msg = "Highly relevant to reference"
                score = 0.95
            elif similarity > 0.6:
                feedback_msg = "Substantially relevant to reference"
                score = 0.80
            elif similarity > 0.4:
                feedback_msg = "Moderately relevant to reference"
                score = 0.60
                issues.append("Content diverges from reference topic")
            else:
                feedback_msg = "Marginally relevant to reference"
                score = 0.40
                issues.append("Content significantly diverges from reference")
        
        feedback = f"Relevance assessment: {feedback_msg if reference else 'No reference provided'}"
        result = ValidationResult(
            criterion=ValidationCriteria.RELEVANCE,
            score=score,
            feedback=feedback,
            issues=issues
        )
        
        if self.verbose:
            print(f"Relevance Score: {score:.2f} - {feedback}")
        
        return result
    
    def validate_all(self, content: str, reference: Optional[str] = None) -> Dict[str, ValidationResult]:
        """
        Run all validation checks on content.
        
        Args:
            content: Content to validate
            reference: Optional reference content for comparison
            
        Returns:
            Dictionary of validation results by criterion
        """
        results = {
            ValidationCriteria.ACCURACY.value: self.validate_accuracy(content, reference),
            ValidationCriteria.COMPLETENESS.value: self.validate_completeness(content, reference),
            ValidationCriteria.CLARITY.value: self.validate_clarity(content),
            ValidationCriteria.CONSISTENCY.value: self.validate_consistency(content),
            ValidationCriteria.RELEVANCE.value: self.validate_relevance(content, reference)
        }
        
        self.validation_history.append((content, {k: v.to_dict() for k, v in results.items()}))
        return results
    
    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class AugmentationEngine:
    """
    Augments content to improve quality across multiple dimensions.
    
    Implements various strategies to enhance, expand, simplify, and correct
    generated content.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the augmentation engine.
        
        Args:
            verbose: Enable detailed augmentation feedback
        """
        self.verbose = verbose
        self.augmentation_history: List[AugmentationResult] = []
    
    def improve(self, content: str) -> AugmentationResult:
        """
        Improve content quality through refinement.
        
        Args:
            content: Content to improve
            
        Returns:
            AugmentationResult with improved content
        """
        improved = content
        changes = []
        
        # Remove redundant phrases
        redundant_phrases = [
            (r'\bit is important to note that\b', 'it is important that'),
            (r'\bin conclusion,\s+to summarize', 'in summary'),
            (r'\bvery\s+very\b', 'very'),
            (r'\bactually,\s+in fact', 'in fact'),
        ]
        
        for pattern, replacement in redundant_phrases:
            if re.search(pattern, improved, re.IGNORECASE):
                improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)
                changes.append(f"Removed redundancy: '{pattern}' → '{replacement}'")
        
        # Add better transitions
        transition_improvements = {
            'However': 'Nevertheless',
            'Also': 'Furthermore',
            'But': 'Conversely',
        }
        
        for old_trans, new_trans in transition_improvements.items():
            if old_trans in improved:
                improved = improved.replace(old_trans, new_trans)
                changes.append(f"Enhanced transition: '{old_trans}' → '{new_trans}'")
        
        # Calculate quality improvement
        quality_improvement = self._calculate_improvement_score(content, improved, changes)
        
        result = AugmentationResult(
            strategy=AugmentationStrategy.IMPROVE,
            original_content=content,
            improved_content=improved,
            quality_improvement=quality_improvement,
            changes_made=changes
        )
        
        if self.verbose:
            print(f"Improved content with {len(changes)} refinements (quality +{quality_improvement:.2%})")
        
        self.augmentation_history.append(result)
        return result
    
    def expand(self, content: str, expansion_ratio: float = 1.5) -> AugmentationResult:
        """
        Expand content with additional detail and examples.
        
        Args:
            content: Content to expand
            expansion_ratio: Target ratio of new content length to original
            
        Returns:
            AugmentationResult with expanded content
        """
        expanded = content
        changes = []
        current_ratio = 1.0
        
        # Add example sections
        if "example" not in expanded.lower() and len(expanded.split()) > 50:
            expanded += "\n\nExample: [Consider adding a concrete example here to illustrate the point.]"
            changes.append("Added example section placeholder")
            current_ratio += 0.2
        
        # Add context sentences
        if current_ratio < expansion_ratio:
            sentences = expanded.split('. ')
            for i, sentence in enumerate(sentences[:-1]):
                if len(sentence.split()) > 10:
                    context = f" This is significant because it demonstrates the underlying principles."
                    sentences[i] = sentence + context
                    changes.append(f"Added contextual detail to sentence {i+1}")
                    current_ratio += 0.1
                    if current_ratio >= expansion_ratio:
                        break
            expanded = '. '.join(sentences)
        
        quality_improvement = self._calculate_improvement_score(content, expanded, changes)
        
        result = AugmentationResult(
            strategy=AugmentationStrategy.EXPAND,
            original_content=content,
            improved_content=expanded,
            quality_improvement=quality_improvement,
            changes_made=changes
        )
        
        if self.verbose:
            print(f"Expanded content: {len(content)} → {len(expanded)} chars, {len(changes)} additions")
        
        self.augmentation_history.append(result)
        return result
    
    def simplify(self, content: str) -> AugmentationResult:
        """
        Simplify content for better clarity and accessibility.
        
        Args:
            content: Content to simplify
            
        Returns:
            AugmentationResult with simplified content
        """
        simplified = content
        changes = []
        
        # Replace complex words
        simplification_map = {
            r'\butilize\b': 'use',
            r'\bacquire\b': 'get',
            r'\bfacilitate\b': 'help',
            r'\bnotwithstanding\b': 'despite',
            r'\bconsequently\b': 'so',
            r'\bpertaining to\b': 'about',
            r'\bcommencement\b': 'start',
        }
        
        for complex_word, simple_word in simplification_map.items():
            matches = re.findall(complex_word, simplified, re.IGNORECASE)
            if matches:
                simplified = re.sub(complex_word, simple_word, simplified, flags=re.IGNORECASE)
                changes.append(f"Simplified: '{complex_word}' → '{simple_word}'")
        
        # Break long sentences
        sentences = simplified.split('. ')
        new_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 20:
                # Try to break at conjunctions
                if ' and ' in sentence:
                    parts = sentence.split(' and ')
                    new_sentences.extend([p.strip() for p in parts if p.strip()])
                    changes.append("Broke long sentence at conjunction")
                else:
                    new_sentences.append(sentence)
            else:
                new_sentences.append(sentence)
        
        simplified = '. '.join(new_sentences)
        
        quality_improvement = self._calculate_improvement_score(content, simplified, changes)
        
        result = AugmentationResult(
            strategy=AugmentationStrategy.SIMPLIFY,
            original_content=content,
            improved_content=simplified,
            quality_improvement=quality_improvement,
            changes_made=changes
        )
        
        if self.verbose:
            print(f"Simplified content: {len(changes)} simplifications, quality {quality_improvement:.2%}")
        
        self.augmentation_history.append(result)
        return result
    
    def correct(self, content: str) -> AugmentationResult:
        """
        Correct errors and inconsistencies in content.
        
        Args:
            content: Content to correct
            
        Returns:
            AugmentationResult with corrected content
        """
        corrected = content
        changes = []
        
        # Fix common grammar errors
        corrections = {
            r'\bthere is\s+multiple\b': 'there are multiple',
            r'\bthe data\s+shows\b': 'the data show',
            r'\btrends\s+indicates\b': 'trends indicate',
            r'\bis\s+your\b': 'are your',
        }
        
        for error_pattern, fix in corrections.items():
            if re.search(error_pattern, corrected, re.IGNORECASE):
                corrected = re.sub(error_pattern, fix, corrected, flags=re.IGNORECASE)
                changes.append(f"Corrected grammar: '{error_pattern}' → '{fix}'")
        
        # Fix capitalization issues
        corrected = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), corrected, flags=re.MULTILINE)
        if re.search(r'^[a-z]', corrected):
            changes.append("Fixed capitalization at sentence starts")
        
        # Remove extra whitespace
        corrected = re.sub(r'\s+', ' ', corrected)
        corrected = re.sub(r'\s+\.', '.', corrected)
        if corrected != content:
            changes.append("Removed extra whitespace")
        
        quality_improvement = self._calculate_improvement_score(content, corrected, changes)
        
        result = AugmentationResult(
            strategy=AugmentationStrategy.CORRECT,
            original_content=content,
            improved_content=corrected,
            quality_improvement=quality_improvement,
            changes_made=changes
        )
        
        if self.verbose:
            print(f"Corrected content: {len(changes)} corrections applied")
        
        self.augmentation_history.append(result)
        return result
    
    def clarify(self, content: str) -> AugmentationResult:
        """
        Clarify content for improved understanding.
        
        Args:
            content: Content to clarify
            
        Returns:
            AugmentationResult with clarified content
        """
        clarified = content
        changes = []
        
        # Add clarifying phrases
        ambiguous_terms = {
            r'\bit\b': 'this concept',
            r'\bthat\b(?!\s+is)': 'that specific item',
        }
        
        # Be conservative with replacements to avoid over-clarifying
        sentences = clarified.split('. ')
        for i, sentence in enumerate(sentences):
            if 'this is important' in sentence.lower():
                sentences[i] = sentence.replace('this is important', 'this critical point')
                changes.append(f"Clarified importance marker in sentence {i+1}")
        
        clarified = '. '.join(sentences)
        
        # Add transition words for clarity
        if not any(trans in clarified for trans in ['therefore', 'thus', 'hence', 'consequently']):
            # This is optional - only add if content is complex
            if len(clarified.split()) > 100:
                clarified = clarified.replace('As a result', 'Consequently')
                changes.append("Added clarity transition words")
        
        quality_improvement = self._calculate_improvement_score(content, clarified, changes)
        
        result = AugmentationResult(
            strategy=AugmentationStrategy.CLARIFY,
            original_content=content,
            improved_content=clarified,
            quality_improvement=quality_improvement,
            changes_made=changes
        )
        
        if self.verbose:
            print(f"Clarified content: {len(changes)} clarifications")
        
        self.augmentation_history.append(result)
        return result
    
    @staticmethod
    def _calculate_improvement_score(original: str, improved: str, changes: List[str]) -> float:
        """Calculate quality improvement score."""
        if not changes:
            return 0.0
        
        # Base improvement from number of changes
        improvement = min(0.3, len(changes) * 0.05)
        
        # Bonus for reducing length while maintaining info (conciseness)
        if len(improved) < len(original):
            compression_ratio = 1.0 - (len(improved) / len(original))
            improvement += min(0.2, compression_ratio * 0.3)
        
        # Bonus for increasing clarity (sentence count should be higher for long content)
        original_sentences = len([s for s in original.split('.') if s.strip()])
        improved_sentences = len([s for s in improved.split('.') if s.strip()])
        
        if improved_sentences > original_sentences:
            improvement += min(0.2, (improved_sentences - original_sentences) * 0.05)
        
        return min(1.0, improvement)


class CAGTool(BaseTool):
    """
    Corrective Augmented Generation Tool for CrewAI.
    
    A comprehensive tool that validates content across multiple criteria
    and iteratively improves it through targeted augmentation strategies.
    
    Attributes:
        name: Tool identifier
        description: Tool functionality description
        max_iterations: Maximum improvement iterations per run
        quality_threshold: Target quality score (0.0-1.0)
        auto_improve: Automatically improve if below threshold
    """
    
    name: str = "CAGTool"
    description: str = "Corrective Augmented Generation tool for validating and improving content quality through multi-criteria assessment and iterative augmentation."
    
    # Pydantic-compatible attributes
    max_iterations: int = 3
    quality_threshold: float = 0.75
    auto_improve: bool = True
    verbose: bool = False
    validator: Any = None
    augmenter: Any = None
    results_history: List = []
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        max_iterations: int = 3,
        quality_threshold: float = 0.75,
        auto_improve: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the CAG tool.
        
        Args:
            max_iterations: Maximum iterations for improvement
            quality_threshold: Target quality score (0.0-1.0)
            auto_improve: Enable automatic improvement mode
            verbose: Enable detailed output
        """
        super().__init__(
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            auto_improve=auto_improve,
            verbose=verbose,
            validator=CorrectivenessValidator(verbose=verbose),
            augmenter=AugmentationEngine(verbose=verbose),
            results_history=[]
        )
    
    def _run(
        self,
        content: str,
        reference: Optional[str] = None,
        auto_improve: Optional[bool] = None,
        strategies: Optional[List[str]] = None,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Execute the CAG process on content.
        
        Args:
            content: Content to validate and improve
            reference: Optional reference content for comparison
            auto_improve: Override auto_improve setting
            strategies: List of augmentation strategies to use
            verbose: Override verbose setting
            
        Returns:
            Dictionary with validation and augmentation results
        """
        if verbose is None:
            verbose = self.verbose
        if auto_improve is None:
            auto_improve = self.auto_improve
        
        current_content = content
        iteration = 0
        validation_results = {}
        augmentation_history = []
        
        if verbose:
            print(f"Starting CAG process on {len(content)} chars of content...")
        
        # Initial validation
        validation_results = self.validator.validate_all(current_content, reference)
        quality_score = self._calculate_overall_quality(validation_results)
        
        if verbose:
            print(f"Initial quality score: {quality_score:.2%}")
        
        # Iterative improvement if enabled
        if auto_improve and quality_score < self.quality_threshold:
            while iteration < self.max_iterations:
                iteration += 1
                
                if verbose:
                    print(f"\nImprovement iteration {iteration}/{self.max_iterations}...")
                
                # Select strategy based on lowest-scoring criterion
                strategy = self._select_strategy(validation_results, strategies)
                
                # Apply augmentation
                aug_result = self._apply_augmentation(current_content, strategy)
                augmentation_history.append(aug_result)
                
                current_content = aug_result.improved_content
                
                # Re-validate
                validation_results = self.validator.validate_all(current_content, reference)
                new_quality = self._calculate_overall_quality(validation_results)
                
                if verbose:
                    improvement = new_quality - quality_score
                    print(f"Quality after {strategy.value}: {new_quality:.2%} (change: {improvement:+.2%})")
                
                # Stop if quality threshold reached
                if new_quality >= self.quality_threshold:
                    quality_score = new_quality
                    if verbose:
                        print(f"Reached quality threshold at iteration {iteration}")
                    break
                
                # Stop if no improvement
                if new_quality <= quality_score:
                    if verbose:
                        print("No quality improvement in this iteration, stopping")
                    break
                
                quality_score = new_quality
        
        # Create result
        cag_result = CAGResult(
            original_content=content,
            final_content=current_content,
            quality_score=quality_score,
            validation_results=validation_results,
            augmentation_history=augmentation_history,
            iterations_used=iteration,
            reference_content=reference
        )
        
        self.results_history.append(cag_result)
        
        # Return formatted result
        return {
            "original_content": content,
            "final_content": current_content,
            "quality_score": quality_score,
            "iterations_used": iteration,
            "validation_scores": {
                k: v.score for k, v in validation_results.items()
            },
            "validation_feedback": {
                k: v.feedback for k, v in validation_results.items()
            },
            "changes_made": [change for aug in augmentation_history for change in aug.changes_made],
            "reference_used": reference is not None
        }
    
    def _select_strategy(
        self,
        validation_results: Dict[str, ValidationResult],
        user_strategies: Optional[List[str]] = None
    ) -> AugmentationStrategy:
        """
        Select augmentation strategy based on validation results.
        
        Args:
            validation_results: Current validation results
            user_strategies: User-specified strategies to choose from
            
        Returns:
            Selected AugmentationStrategy
        """
        if user_strategies:
            return AugmentationStrategy[user_strategies[0].upper()]
        
        # Select strategy targeting weakest criterion
        scores = {k: v.score for k, v in validation_results.items()}
        weakest = min(scores, key=scores.get)
        
        strategy_mapping = {
            'accuracy': AugmentationStrategy.CORRECT,
            'completeness': AugmentationStrategy.EXPAND,
            'clarity': AugmentationStrategy.CLARIFY,
            'consistency': AugmentationStrategy.CORRECT,
            'relevance': AugmentationStrategy.IMPROVE,
        }
        
        return strategy_mapping.get(weakest, AugmentationStrategy.IMPROVE)
    
    def _apply_augmentation(
        self,
        content: str,
        strategy: AugmentationStrategy
    ) -> AugmentationResult:
        """
        Apply specified augmentation strategy.
        
        Args:
            content: Content to augment
            strategy: Augmentation strategy to apply
            
        Returns:
            AugmentationResult with augmented content
        """
        if strategy == AugmentationStrategy.IMPROVE:
            return self.augmenter.improve(content)
        elif strategy == AugmentationStrategy.EXPAND:
            return self.augmenter.expand(content)
        elif strategy == AugmentationStrategy.SIMPLIFY:
            return self.augmenter.simplify(content)
        elif strategy == AugmentationStrategy.CORRECT:
            return self.augmenter.correct(content)
        elif strategy == AugmentationStrategy.CLARIFY:
            return self.augmenter.clarify(content)
        else:
            return self.augmenter.improve(content)
    
    @staticmethod
    def _calculate_overall_quality(validation_results: Dict[str, ValidationResult]) -> float:
        """
        Calculate overall quality score from validation results.
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            Overall quality score (0.0-1.0)
        """
        if not validation_results:
            return 0.0
        
        # Weighted average of criteria
        weights = {
            'accuracy': 0.25,
            'completeness': 0.20,
            'clarity': 0.20,
            'consistency': 0.20,
            'relevance': 0.15,
        }
        
        total_score = 0.0
        for criterion, result in validation_results.items():
            weight = weights.get(criterion, 1.0 / len(validation_results))
            total_score += result.score * weight
        
        return total_score
    
    def get_results_summary(self, result_index: int = -1) -> Dict[str, Any]:
        """
        Get summary of CAG results.
        
        Args:
            result_index: Index of result to summarize (default: latest)
            
        Returns:
            Summary dictionary
        """
        if not self.results_history:
            return {"error": "No results available"}
        
        result = self.results_history[result_index]
        
        return {
            "quality_improvement": result.quality_score,
            "iterations_used": result.iterations_used,
            "original_length": len(result.original_content),
            "final_length": len(result.final_content),
            "validation_summary": {k: v.score for k, v in result.validation_results.items()},
            "total_changes": sum(len(a.changes_made) for a in result.augmentation_history),
            "strategies_used": [a.strategy.value for a in result.augmentation_history]
        }
    
    def save_results(self, filepath: str, result_index: int = -1):
        """
        Save CAG results to JSON file.
        
        Args:
            filepath: Path to save results
            result_index: Index of result to save (default: latest)
        """
        if not self.results_history:
            raise ValueError("No results to save")
        
        result = self.results_history[result_index]
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def get_improvement_analysis(self) -> Dict[str, Any]:
        """
        Analyze improvement patterns across all results.
        
        Returns:
            Analysis dictionary with patterns and statistics
        """
        if not self.results_history:
            return {"error": "No results for analysis"}
        
        analysis = {
            "total_runs": len(self.results_history),
            "average_quality_improvement": sum(r.quality_score for r in self.results_history) / len(self.results_history),
            "most_used_strategy": self._get_most_used_strategy(),
            "average_iterations": sum(r.iterations_used for r in self.results_history) / len(self.results_history),
            "total_content_processed": sum(len(r.original_content) for r in self.results_history)
        }
        
        return analysis
    
    def _get_most_used_strategy(self) -> str:
        """Get most frequently used augmentation strategy."""
        strategy_counts = {}
        
        for result in self.results_history:
            for aug in result.augmentation_history:
                strategy_counts[aug.strategy.value] = strategy_counts.get(aug.strategy.value, 0) + 1
        
        if not strategy_counts:
            return "none"
        
        return max(strategy_counts, key=strategy_counts.get)
