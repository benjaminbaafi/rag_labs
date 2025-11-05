"""Prompt loader utility for loading prompts from JSON file."""
import json
from pathlib import Path
from typing import Optional, Dict, Any


class PromptLoader:
    """Load prompts from JSON file in the prompts directory."""
    
    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize prompt loader.
        
        Args:
            prompts_file: Path to prompts.json file (default: prompts/prompts.json)
        """
        if prompts_file is None:
            # Get prompts directory relative to this file
            current_dir = Path(__file__).parent
            self.prompts_file = current_dir / "prompts.json"
        else:
            self.prompts_file = Path(prompts_file)
        
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from JSON file."""
        if not self.prompts_file.exists():
            raise FileNotFoundError(
                f"Prompts file not found: {self.prompts_file}\n"
                f"Please create prompts.json in the prompts directory."
            )
        
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            self._prompts = json.load(f)
    
    def load(self, prompt_path: str) -> str:
        """
        Load a prompt from the JSON structure.
        
        Args:
            prompt_path: Path to prompt in JSON (e.g., "system.default", "user.rag")
            
        Returns:
            Prompt text as string
        """
        keys = prompt_path.split('.')
        value = self._prompts
        
        try:
            for key in keys:
                value = value[key]
            
            if not isinstance(value, str):
                raise ValueError(f"Prompt at '{prompt_path}' is not a string")
            
            return value.strip()
        except KeyError as e:
            available = self.list_available()
            raise FileNotFoundError(
                f"Prompt not found: {prompt_path}\n"
                f"Available prompts: {available}"
            ) from e
    
    def load_template(self, template_path: str, **kwargs) -> str:
        """
        Load a prompt template and format it with variables.
        
        Args:
            template_path: Path to template in JSON (e.g., "user.rag", "multi_step.subqueries")
            **kwargs: Variables to format the template with
            
        Returns:
            Formatted prompt text
        """
        template = self.load(template_path)
        return template.format(**kwargs)
    
    def list_available(self) -> list:
        """List all available prompts in a flat structure."""
        def _flatten(obj, prefix=""):
            """Flatten nested dictionary to dot-notation paths."""
            result = []
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    result.extend(_flatten(value, path))
                else:
                    result.append(path)
            return result
        
        return sorted(_flatten(self._prompts))
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire prompts structure."""
        return self._prompts.copy()


# Global prompt loader instance
_loader: Optional[PromptLoader] = None


def _get_loader() -> PromptLoader:
    """Get or create the global prompt loader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


def load_prompt(prompt_path: str) -> str:
    """
    Load a prompt by path (convenience function).
    
    Args:
        prompt_path: Path to prompt in JSON (e.g., "system.default", "user.rag")
        
    Returns:
        Prompt text as string
    """
    return _get_loader().load(prompt_path)


def load_template(template_path: str, **kwargs) -> str:
    """
    Load and format a prompt template (convenience function).
    
    Args:
        template_path: Path to template in JSON (e.g., "user.rag", "multi_step.subqueries")
        **kwargs: Variables to format the template with
        
    Returns:
        Formatted prompt text
    """
    return _get_loader().load_template(template_path, **kwargs)

