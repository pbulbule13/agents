import requests
from typing import Any, Dict, Optional

class EuriaiN8N:
    """
    Wrapper for n8n workflow automation integration in the EURI SDK.
    Allows triggering n8n workflows and exchanging data via REST API.
    
    Template Files:
    - n8n_workflow_template.json: Complete workflow template for Euri API integration
    - N8N_WORKFLOW_GUIDE.md: Comprehensive setup and usage guide
    - n8n_example_usage.py: Python example for interacting with the workflow
    """
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the n8n wrapper.
        Args:
            base_url: Base URL of the n8n instance (e.g., http://localhost:5678 or cloud URL)
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    def trigger_workflow(self, workflow_id: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trigger an n8n workflow by ID, optionally passing data.
        Returns the workflow execution response.
        """
        url = f"{self.base_url}/webhook/{workflow_id}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(url, json=data or {}, headers=headers)
        response.raise_for_status()
        return response.json()

    def chat_completion(
        self,
        webhook_id: str,
        prompt: str,
        euri_api_key: str,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Convenient method to send chat completion requests to the Euri API workflow.
        
        Args:
            webhook_id: The webhook ID or path for the Euri chat workflow
            prompt: The message to send to the AI
            euri_api_key: Your Euri API key
            model: The model to use (default: gpt-4.1-nano)
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing the API response
        """
        data = {
            "api_key": euri_api_key,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return self.trigger_workflow(webhook_id, data)

    def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow execution.
        
        Args:
            execution_id: The execution ID returned from trigger_workflow
            
        Returns:
            Dictionary containing execution status
        """
        url = f"{self.base_url}/api/v1/executions/{execution_id}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_template_info() -> Dict[str, str]:
        """
        Get information about available template files.
        
        Returns:
            Dictionary with template file descriptions
        """
        return {
            "workflow_template": "n8n_workflow_template.json - Complete n8n workflow for Euri API integration",
            "setup_guide": "N8N_WORKFLOW_GUIDE.md - Comprehensive setup and usage documentation",
            "python_example": "n8n_example_usage.py - Python client example for workflow interaction",
            "description": "Ready-to-use n8n workflow template for Euri AI chat completions with any supported model"
        } 