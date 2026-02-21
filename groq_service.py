"""
Groq LLM Service for VQA Accessibility
Generates descriptive 2-sentence narrations for blind users
"""
import os
from typing import Dict, Optional
from groq import Groq
class GroqDescriptionService:
    """Service to generate accessible descriptions using Groq LLM"""
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq service
        Args:
            api_key: Groq API key (if not provided, reads from GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter"
            )
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
    def generate_description(
        self, 
        question: str, 
        answer: str,
        max_retries: int = 2
    ) -> Dict[str, str]:
        """
        Generate a 2-sentence accessible description for blind users
        Args:
            question: The question asked by the user
            answer: The VQA model's answer
            max_retries: Number of retry attempts on failure
        Returns:
            Dict with 'description' and 'status' keys
        """
        prompt = f"""You are an accessibility assistant helping blind users understand visual question answering results.
Question asked: "{question}"
Answer from VQA model: "{answer}"
Task: Create a clear, natural 2-sentence description that:
1. First sentence: Restates the question and provides the answer
2. Second sentence: Adds helpful context or clarification
Keep it concise, natural, and easy to understand when spoken aloud.
Example:
Question: "What color is the car?"
Answer: "red"
Description: "The question asks about the color of the car, and the answer is red. This indicates there is a red-colored vehicle visible in the image."
Now generate the description:"""
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful accessibility assistant. Always respond with exactly 2 clear, natural sentences."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=150,
                    top_p=0.9
                )
                description = response.choices[0].message.content.strip()
                if description.startswith("Description:"):
                    description = description.replace("Description:", "").strip()
                return {
                    "description": description,
                    "status": "success",
                    "model": self.model
                }
            except Exception as e:
                if attempt < max_retries:
                    continue
                else:
                    fallback = f"The question asks: {question}. The answer is: {answer}."
                    return {
                        "description": fallback,
                        "status": "fallback",
                        "error": str(e)
                    }
    def generate_batch_descriptions(
        self, 
        qa_pairs: list[Dict[str, str]]
    ) -> list[Dict[str, str]]:
        """
        Generate descriptions for multiple Q&A pairs
        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys
        Returns:
            List of description results
        """
        results = []
        for pair in qa_pairs:
            result = self.generate_description(
                question=pair.get("question", ""),
                answer=pair.get("answer", "")
            )
            results.append(result)
        return results
_groq_service_instance = None
def get_groq_service(api_key: Optional[str] = None) -> GroqDescriptionService:
    """
    Get or create Groq service singleton
    Args:
        api_key: Optional API key (uses env var if not provided)
    Returns:
        GroqDescriptionService instance
    """
    global _groq_service_instance
    if _groq_service_instance is None:
        _groq_service_instance = GroqDescriptionService(api_key=api_key)
    return _groq_service_instance