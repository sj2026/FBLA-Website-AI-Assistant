from requests import post as rpost
from langchain_core.language_models.llms import LLM


class LLaMa(LLM):
    
    """
        Wrapper class to invoke LLM
    """
    def _call(self, prompt, **kwargs):
        return self.call_llama(prompt)

    @property
    def _llm_type(self):
        return "llama-3.2-3b"
    
    def call_llama(self,prompt):

        """
        Invokes the llm
        Uses the llama model in Ollama environment

        Args:
            prompt: prompt to LLM

        Returns:
            response from LLM.

        """
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
        }

        response = rpost(
            "http://localhost:11434/api/generate",
            headers=headers,
            json=payload
        )
        return response.json()["response"]
