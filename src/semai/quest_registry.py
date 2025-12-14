"""
Side Quest Registry - Central repository for managing consistent model benchmarking quests
Provides persistent storage and retrieval of quest definitions and results
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QuestRegistry:
    """Central registry for managing side quests"""
    
    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize quest registry
        
        Args:
            registry_dir: Directory to store quest data (default: .quests in project root)
        """
        if registry_dir is None:
            registry_dir = Path(__file__).parent.parent.parent / '.quests'
        
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.quests_dir = self.registry_dir / 'definitions'
        self.results_dir = self.registry_dir / 'results'
        self.sets_dir = self.registry_dir / 'sets'
        
        for d in [self.quests_dir, self.results_dir, self.sets_dir]:
            d.mkdir(exist_ok=True)
        
        logger.info(f"Quest registry initialized at {self.registry_dir}")
    
    def register_quest(self, quest_data: Dict[str, Any]) -> str:
        """
        Register a new quest in the registry
        
        Args:
            quest_data: Quest definition dictionary
            
        Returns:
            str: Quest ID
        """
        quest_id = quest_data.get('quest_id')
        if not quest_id:
            raise ValueError("quest_data must include 'quest_id'")
        
        quest_file = self.quests_dir / f"{quest_id}.json"
        
        with open(quest_file, 'w') as f:
            json.dump(quest_data, f, indent=2)
        
        logger.info(f"Registered quest: {quest_id}")
        return quest_id
    
    def get_quest(self, quest_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a quest definition"""
        quest_file = self.quests_dir / f"{quest_id}.json"
        
        if not quest_file.exists():
            return None
        
        with open(quest_file, 'r') as f:
            return json.load(f)
    
    def list_quests(self, quest_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered quests, optionally filtered by type"""
        quests = []
        
        for quest_file in self.quests_dir.glob("*.json"):
            with open(quest_file, 'r') as f:
                quest = json.load(f)
            
            if quest_type is None or quest.get('quest_type') == quest_type:
                quests.append(quest)
        
        return quests
    
    def log_result(self, quest_id: str, model_name: str, result_data: Dict[str, Any]) -> str:
        """
        Log quest result for a model
        
        Args:
            quest_id: Quest ID
            model_name: Model being tested
            result_data: Result metrics and metadata
            
        Returns:
            str: Result file ID
        """
        result_id = f"{quest_id}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_file = self.results_dir / f"{result_id}.json"
        
        result_record = {
            'result_id': result_id,
            'quest_id': quest_id,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            **result_data
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_record, f, indent=2)
        
        logger.info(f"Logged result: {result_id}")
        return result_id
    
    def get_results(self, quest_id: str, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve results for a quest
        
        Args:
            quest_id: Quest ID
            model_name: Optional filter for specific model
            
        Returns:
            List of result records
        """
        results = []
        
        for result_file in self.results_dir.glob(f"{quest_id}_*.json"):
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            if model_name is None or result.get('model_name') == model_name:
                results.append(result)
        
        return results
    
    def register_quest_set(self, set_data: Dict[str, Any]) -> str:
        """
        Register a set of related quests
        
        Args:
            set_data: Quest set definition
            
        Returns:
            str: Set ID
        """
        set_id = f"SET_{set_data.get('theme', 'custom')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        set_file = self.sets_dir / f"{set_id}.json"
        
        set_record = {
            'set_id': set_id,
            'created_at': datetime.now().isoformat(),
            **set_data
        }
        
        with open(set_file, 'w') as f:
            json.dump(set_record, f, indent=2)
        
        logger.info(f"Registered quest set: {set_id}")
        return set_id
    
    def get_quest_set(self, set_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a quest set"""
        set_file = self.sets_dir / f"{set_id}.json"
        
        if not set_file.exists():
            return None
        
        with open(set_file, 'r') as f:
            return json.load(f)
    
    def generate_leaderboard(self, metric: str = 'accuracy', limit: int = 10) -> Dict[str, Any]:
        """
        Generate leaderboard of model performance
        
        Args:
            metric: Metric to rank by
            limit: Top N results
            
        Returns:
            Leaderboard data
        """
        model_scores = {}
        
        for result_file in self.results_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            model_name = result.get('model_name')
            metrics = result.get('metrics', {})
            score = metrics.get(metric, 0)
            
            if model_name not in model_scores:
                model_scores[model_name] = []
            
            model_scores[model_name].append(score)
        
        # Calculate average scores
        leaderboard = [
            {'model': model, 'avg_score': sum(scores) / len(scores), 'n_quests': len(scores)}
            for model, scores in model_scores.items()
        ]
        
        # Sort by average score (descending)
        leaderboard.sort(key=lambda x: x['avg_score'], reverse=True)
        
        return {
            'metric': metric,
            'timestamp': datetime.now().isoformat(),
            'leaderboard': leaderboard[:limit]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        quests = list(self.quests_dir.glob("*.json"))
        results = list(self.results_dir.glob("*.json"))
        sets = list(self.sets_dir.glob("*.json"))
        
        return {
            'total_quests': len(quests),
            'total_results': len(results),
            'total_sets': len(sets),
            'registry_path': str(self.registry_dir)
        }


def get_quest_registry(registry_dir: Optional[Path] = None) -> QuestRegistry:
    """Factory function to get quest registry instance"""
    return QuestRegistry(registry_dir)
