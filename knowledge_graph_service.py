"""
Knowledge Graph Service for Neuro-Symbolic VQA
Uses ConceptNet API to provide common-sense reasoning capabilities
"""

import requests
import re
from typing import Dict, List, Optional
from functools import lru_cache
import time


class KnowledgeGraphService:
    """
    Lightweight ConceptNet API wrapper for common-sense reasoning.
    Enhances VQA answers with external knowledge about object properties,
    capabilities, uses, and relationships.
    """
    
    CONCEPTNET_API = "https://api.conceptnet.io"
    
    # Common-sense question patterns
    COMMONSENSE_PATTERNS = [
        # Capability questions
        (r'can .* (melt|freeze|fly|swim|float|sink|break|burn|explode)', 'CapableOf'),
        (r'is .* able to', 'CapableOf'),
        (r'does .* (float|sink)', 'CapableOf'),
        
        # Property questions
        (r'is .* (edible|poisonous|dangerous|safe|hot|cold|sweet|sour)', 'HasProperty'),
        (r'is this (food|drink|toy|tool|weapon)', 'HasProperty'),
        
        # Purpose questions
        (r'what .* (used for|for)', 'UsedFor'),
        (r'why .* (used|made)', 'UsedFor'),
        (r'how .* use', 'UsedFor'),
        
        # Composition questions
        (r'what .* made (of|from)', 'MadeOf'),
        (r'what .* (material|ingredient)', 'MadeOf'),
        
        # Location questions
        (r'where .* (found|located|kept|stored)', 'AtLocation'),
        (r'where (do|does) .* (live|grow)', 'AtLocation'),
    ]
    
    def __init__(self, cache_size=100, timeout=5):
        """
        Initialize Knowledge Graph service.
        
        Args:
            cache_size: Number of API responses to cache
            timeout: API request timeout in seconds
        """
        self.timeout = timeout
        self.cache_size = cache_size
        print("✅ Knowledge Graph service initialized (ConceptNet API)")
    
    @lru_cache(maxsize=100)
    def _query_conceptnet(self, concept: str, relation: str, limit: int = 10) -> Optional[Dict]:
        """
        Query ConceptNet API with caching.
        
        Args:
            concept: Concept to query (e.g., "ice_cream")
            relation: Relation type (e.g., "CapableOf", "HasProperty")
            limit: Maximum number of results
            
        Returns:
            API response dict or None if failed
        """
        try:
            # Normalize concept (replace spaces with underscores)
            concept = concept.lower().replace(' ', '_')
            
            # Build API URL
            url = f"{self.CONCEPTNET_API}/query"
            params = {
                'start': f'/c/en/{concept}',
                'rel': f'/r/{relation}',
                'limit': limit
            }
            
            # Make request
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            print(f"⚠️  ConceptNet API timeout for {concept}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️  ConceptNet API error: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error querying ConceptNet: {e}")
            return None
    
    def get_concept_properties(self, concept: str) -> Dict[str, List[str]]:

        properties = {
            'CapableOf': [],
            'HasProperty': [],
            'UsedFor': [],
            'MadeOf': [],
            'AtLocation': []
        }
        
        # Query each relation type
        for relation in properties.keys():
            data = self._query_conceptnet(concept, relation)
            
            if data and 'edges' in data:
                for edge in data['edges']:
                    # Extract the end concept
                    if 'end' in edge and 'label' in edge['end']:
                        end_label = edge['end']['label']
                        properties[relation].append(end_label)
        
        return properties
    
    def is_commonsense_question(self, question: str) -> bool:
        """
        Detect if a question requires common-sense reasoning.
        
        Args:
            question: Question string
            
        Returns:
            True if question needs external knowledge
        """
        q_lower = question.lower()
        
        for pattern, _ in self.COMMONSENSE_PATTERNS:
            if re.search(pattern, q_lower):
                return True
        
        return False
    
    def _detect_question_type(self, question: str) -> Optional[str]:
        """
        Detect which ConceptNet relation the question is asking about.
        
        Args:
            question: Question string
            
        Returns:
            Relation type or None
        """
        q_lower = question.lower()
        
        for pattern, relation in self.COMMONSENSE_PATTERNS:
            if re.search(pattern, q_lower):
                return relation
        
        return None
    
    def answer_commonsense_question(self, object_name: str, question: str) -> Optional[str]:
        """
        Answer a common-sense question using Knowledge Graph.
        
        Args:
            object_name: Object detected by VQA (e.g., "ice cream")
            question: User's question
            
        Returns:
            Enhanced answer string or None
        """
        # Detect question type
        relation = self._detect_question_type(question)
        if not relation:
            return None
        
        # Query ConceptNet
        data = self._query_conceptnet(object_name, relation, limit=5)
        if not data or 'edges' not in data:
            return None
        
        # Extract relevant knowledge
        knowledge = []
        for edge in data['edges']:
            if 'end' in edge and 'label' in edge['end']:
                knowledge.append(edge['end']['label'])
        
        if not knowledge:
            return None
        
        # Generate natural language answer based on question type
        return self._synthesize_answer(object_name, question, relation, knowledge)
    
    def _synthesize_answer(self, object_name: str, question: str, 
                          relation: str, knowledge: List[str]) -> str:
        """
        Synthesize natural language answer from knowledge.
        
        Args:
            object_name: Detected object
            question: Original question
            relation: Relation type
            knowledge: List of related concepts from KG
            
        Returns:
            Natural language answer
        """
        q_lower = question.lower()
        
        # Capability questions (can X do Y?)
        if relation == 'CapableOf':
            # Check if specific capability is mentioned
            for capability in knowledge:
                if capability in q_lower:
                    return f"Yes, {object_name} can {capability}."
            
            # General capability answer
            if knowledge:
                caps = ', '.join(knowledge[:3])
                return f"{object_name.capitalize()} can {caps}."
        
        # Property questions (is X Y?)
        elif relation == 'HasProperty':
            # Check for specific property
            if 'edible' in q_lower:
                if 'edible' in knowledge:
                    return f"Yes, {object_name} is edible."
                else:
                    return f"No, {object_name} is not edible."
            
            if 'dangerous' in q_lower or 'safe' in q_lower:
                if any(prop in knowledge for prop in ['dangerous', 'harmful', 'poisonous']):
                    return f"Caution: {object_name} may be dangerous."
                else:
                    return f"{object_name.capitalize()} is generally safe."
            
            # General properties
            if knowledge:
                props = ', '.join(knowledge[:3])
                return f"{object_name.capitalize()} is {props}."
        
        # Purpose questions (what is X used for?)
        elif relation == 'UsedFor':
            if knowledge:
                uses = ', '.join(knowledge[:3])
                return f"{object_name.capitalize()} is used for {uses}."
        
        # Composition questions (what is X made of?)
        elif relation == 'MadeOf':
            if knowledge:
                materials = ', '.join(knowledge[:3])
                return f"{object_name.capitalize()} is made of {materials}."
        
        # Location questions (where is X found?)
        elif relation == 'AtLocation':
            if knowledge:
                locations = ', '.join(knowledge[:2])
                return f"{object_name.capitalize()} is typically found at {locations}."
        
        return None


# Test function
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing Knowledge Graph Service")
    print("=" * 80)
    
    kg = KnowledgeGraphService()
    
    # Test cases
    test_cases = [
        ("ice cream", "Can this melt?"),
        ("apple", "Is this edible?"),
        ("hammer", "What is this used for?"),
        ("knife", "Is this dangerous?"),
        ("bread", "What is this made of?"),
    ]
    
    for obj, question in test_cases:
        print(f"\n📝 Object: {obj}")
        print(f"❓ Question: {question}")
        
        # Check if common-sense question
        is_cs = kg.is_commonsense_question(question)
        print(f"🔍 Common-sense: {is_cs}")
        
        if is_cs:
            # Get answer
            answer = kg.answer_commonsense_question(obj, question)
            print(f"💬 Answer: {answer}")
        
        print("-" * 80)
