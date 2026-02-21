"""
LLM Reasoning Service for VQA
Uses Groq LLM for Chain-of-Thought reasoning instead of hardcoded rules
"""
import os
from typing import Dict, List, Optional, Any
from groq import Groq
import json
class LLMReasoningService:
    """
    Service that uses Groq LLM for deductive reasoning from Wikidata facts.
    Replaces hardcoded if/else rules with flexible Chain-of-Thought reasoning.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize LLM Reasoning service
        Args:
            api_key: Groq API key (if not provided, reads from GROQ_API_KEY env var)
            model: Groq model to use for reasoning
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter"
            )
        self.client = Groq(api_key=self.api_key)
        self.model = model
        print(f"✅ LLM Reasoning Service initialized (model: {model})")
    def reason_with_facts(
        self,
        object_name: str,
        facts: Dict[str, Any],
        question: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Use LLM to reason about a question using Wikidata facts.
        Args:
            object_name: Name of the detected object (e.g., "candle")
            facts: Dictionary of Wikidata facts about the object
            question: User's question
            max_retries: Number of retry attempts on failure
        Returns:
            Dict with 'answer', 'reasoning_chain', and 'confidence' keys
        Example:
            >>> service.reason_with_facts(
            ...     object_name="candle",
            ...     facts={"materials": ["wax"], "categories": ["light source"]},
            ...     question="Can this melt?"
            ... )
            {
                'answer': 'Yes, the candle can melt because it is made of wax...',
                'reasoning_chain': [
                    'The object is a candle',
                    'It is made of wax',
                    'Wax has a low melting point',
                    'Therefore, yes, it can melt'
                ],
                'confidence': 0.95
            }
        """
        prompt = self._build_reasoning_prompt(object_name, facts, question)
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert reasoning assistant for a Visual Question Answering system.
Your task is to use Chain-of-Thought reasoning to answer questions about objects based on factual knowledge.
IMPORTANT: Respond in JSON format with this structure:
{
  "reasoning_chain": ["step 1", "step 2", "step 3"],
  "answer": "final answer in natural language",
  "confidence": 0.0-1.0
}
Keep reasoning steps clear and logical. The answer should be conversational and helpful."""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content.strip()
                result = json.loads(content)
                if not all(key in result for key in ['reasoning_chain', 'answer', 'confidence']):
                    raise ValueError("Invalid response structure from LLM")
                return {
                    'answer': result['answer'],
                    'reasoning_chain': result['reasoning_chain'],
                    'confidence': float(result['confidence']),
                    'status': 'success',
                    'model': self.model
                }
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    continue
                else:
                    return self._fallback_reasoning(object_name, facts, question)
            except Exception as e:
                if attempt < max_retries:
                    continue
                else:
                    print(f"⚠️  LLM reasoning failed: {e}")
                    return self._fallback_reasoning(object_name, facts, question)
    def _build_reasoning_prompt(
        self,
        object_name: str,
        facts: Dict[str, Any],
        question: str
    ) -> str:
        """
        Build a Chain-of-Thought reasoning prompt.
        Args:
            object_name: Name of the object
            facts: Wikidata facts about the object
            question: User's question
        Returns:
            Formatted prompt string
        """
        facts_text = self._format_facts(facts)
        prompt = f"""Question: {question}
Object Detected: {object_name}
Available Facts from Knowledge Graph:
{facts_text}
Task: Use Chain-of-Thought reasoning to answer the question based on the available facts.
Example of good reasoning:
Question: "Can this melt?"
Object: "ice cream"
Facts: {{
  "categories": ["frozen dessert", "food"],
  "materials": ["milk", "sugar", "cream"]
}}
Reasoning:
{{
  "reasoning_chain": [
    "The object is ice cream, which is a frozen dessert",
    "Ice cream is made of milk, sugar, and cream",
    "These ingredients are frozen to create ice cream",
    "Frozen items melt when exposed to heat",
    "Therefore, yes, ice cream can melt at room temperature"
  ],
  "answer": "Yes, ice cream can melt. It's a frozen dessert made from milk, sugar, and cream, which will melt when exposed to temperatures above freezing.",
  "confidence": 0.95
}}
Now reason about the actual question above:"""
        return prompt
    def _format_facts(self, facts: Dict[str, Any]) -> str:
        """Format facts dictionary into readable text."""
        if not facts:
            return "No specific facts available"
        lines = []
        for key, value in facts.items():
            if isinstance(value, list):
                if value:
                    lines.append(f"  - {key}: {', '.join(str(v) for v in value)}")
            elif value:
                lines.append(f"  - {key}: {value}")
        return "\n".join(lines) if lines else "No specific facts available"
    def _fallback_reasoning(
        self,
        object_name: str,
        facts: Dict[str, Any],
        question: str
    ) -> Dict[str, Any]:
        """
        Fallback reasoning when LLM fails.
        Uses simple rule-based approach.
        Args:
            object_name: Name of the object
            facts: Wikidata facts
            question: User's question
        Returns:
            Fallback reasoning result
        """
        q_lower = question.lower()
        if 'melt' in q_lower:
            materials = facts.get('materials', [])
            if any(m in ['wax', 'ice', 'chocolate', 'butter'] for m in materials):
                return {
                    'answer': f"Yes, {object_name} can melt as it contains materials with low melting points.",
                    'reasoning_chain': [
                        f"The {object_name} contains materials that can melt",
                        "These materials have low melting points",
                        "Therefore, it can melt when heated"
                    ],
                    'confidence': 0.7,
                    'status': 'fallback'
                }
        if 'edible' in q_lower or 'eat' in q_lower:
            categories = facts.get('categories', [])
            if any('food' in str(c).lower() for c in categories):
                return {
                    'answer': f"Yes, {object_name} is edible as it is categorized as food.",
                    'reasoning_chain': [
                        f"The {object_name} is categorized as food",
                        "Food items are generally edible",
                        "Therefore, it is edible"
                    ],
                    'confidence': 0.8,
                    'status': 'fallback'
                }
        return {
            'answer': f"Based on the available information about {object_name}, I cannot provide a definitive answer to this question.",
            'reasoning_chain': [
                f"Analyzing {object_name}",
                "Available facts are limited",
                "Cannot make a confident conclusion"
            ],
            'confidence': 0.3,
            'status': 'fallback_generic'
        }
    def batch_reason(
        self,
        reasoning_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform reasoning on multiple tasks.
        Args:
            reasoning_tasks: List of dicts with 'object_name', 'facts', 'question' keys
        Returns:
            List of reasoning results
        """
        results = []
        for task in reasoning_tasks:
            result = self.reason_with_facts(
                object_name=task.get('object_name', ''),
                facts=task.get('facts', {}),
                question=task.get('question', '')
            )
            results.append(result)
        return results
_llm_reasoning_instance = None
def get_llm_reasoning_service(api_key: Optional[str] = None) -> LLMReasoningService:
    """
    Get or create LLM Reasoning service singleton
    Args:
        api_key: Optional API key (uses env var if not provided)
    Returns:
        LLMReasoningService instance
    """
    global _llm_reasoning_instance
    if _llm_reasoning_instance is None:
        _llm_reasoning_instance = LLMReasoningService(api_key=api_key)
    return _llm_reasoning_instance
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Testing LLM Reasoning Service")
    print("=" * 80)
    try:
        service = get_llm_reasoning_service()
        print("\n📝 Test 1: Can a candle melt?")
        result = service.reason_with_facts(
            object_name="candle",
            facts={
                "materials": ["wax", "wick"],
                "categories": ["light source", "household item"],
                "uses": ["provide light", "decoration"]
            },
            question="Can this melt?"
        )
        print(f"Answer: {result['answer']}")
        print(f"Reasoning Chain:")
        for i, step in enumerate(result['reasoning_chain'], 1):
            print(f"  {i}. {step}")
        print(f"Confidence: {result['confidence']}")
        print("\n📝 Test 2: Would ice cream survive in the desert?")
        result = service.reason_with_facts(
            object_name="ice cream",
            facts={
                "materials": ["milk", "sugar", "cream"],
                "categories": ["frozen dessert", "food"],
                "properties": ["cold", "frozen"]
            },
            question="Would this survive in the desert?"
        )
        print(f"Answer: {result['answer']}")
        print(f"Reasoning Chain:")
        for i, step in enumerate(result['reasoning_chain'], 1):
            print(f"  {i}. {step}")
        print(f"Confidence: {result['confidence']}")
        print("\n" + "=" * 80)
        print("✅ Tests completed!")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please set GROQ_API_KEY environment variable")