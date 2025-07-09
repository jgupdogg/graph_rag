"""
Configuration management for GraphRAG workspaces.
Handles workspace initialization, template management, and configuration generation.
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Try to import python-dotenv if available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for GraphRAG workspaces."""
    
    def __init__(self, base_config_dir: Path = None):
        self.base_config_dir = base_config_dir or Path("graphrag_config")
        self._load_environment_variables()
        self.default_settings = self._load_default_settings()
    
    def _load_environment_variables(self):
        """Load environment variables from .env file."""
        env_files = [
            self.base_config_dir / ".env",
            Path(".env"),
            Path("graphrag") / ".env"
        ]
        
        for env_file in env_files:
            if env_file.exists():
                if DOTENV_AVAILABLE:
                    load_dotenv(env_file, override=True)
                    logger.info(f"Loaded environment variables from {env_file}")
                else:
                    # Manual loading if python-dotenv is not available
                    self._load_env_file_manually(env_file)
                break
    
    def _load_env_file_manually(self, env_file: Path):
        """Manually load environment variables from .env file."""
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            logger.info(f"Manually loaded environment variables from {env_file}")
        except Exception as e:
            logger.error(f"Error loading environment variables from {env_file}: {e}")
    
    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default settings from base configuration."""
        settings_path = self.base_config_dir / "settings.yaml"
        if settings_path.exists():
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading default settings: {e}")
        
        # Return minimal default configuration if file doesn't exist
        return self._get_minimal_config()
    
    def _get_minimal_config(self) -> Dict[str, Any]:
        """Get minimal GraphRAG configuration."""
        return {
            "encoding": "utf-8",
            "skip_workflows": [],
            "llm": {
                "api_key": "${GRAPHRAG_API_KEY}",
                "type": "openai_chat",
                "model": "gpt-3.5-turbo",
                "model_supports_json": True,
                "max_tokens": 4000,
                "temperature": 0,
                "request_timeout": 180.0,
                "api_base": None,
                "api_version": None,
                "organization": None,
                "proxy": None,
                "cognitive_services_endpoint": None,
                "deployment_name": None,
                "tokens_per_minute": 0,
                "requests_per_minute": 0,
                "max_retries": 10,
                "max_retry_wait": 10.0,
                "sleep_on_rate_limit_recommendation": True,
                "concurrent_requests": 25
            },
            "parallelization": {
                "stagger": 0.3,
                "num_threads": 50
            },
            "async_mode": "threaded",
            "embeddings": {
                "async_mode": "threaded",
                "llm": {
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "type": "openai_embedding",
                    "model": "text-embedding-3-small",
                    "max_tokens": 8191,
                    "request_timeout": 180.0,
                    "api_base": None,
                    "api_version": None,
                    "organization": None,
                    "proxy": None,
                    "cognitive_services_endpoint": None,
                    "deployment_name": None,
                    "tokens_per_minute": 0,
                    "requests_per_minute": 0,
                    "max_retries": 10,
                    "max_retry_wait": 10.0,
                    "sleep_on_rate_limit_recommendation": True,
                    "concurrent_requests": 25
                },
                "parallelization": {
                    "stagger": 0.3,
                    "num_threads": 50
                }
            },
            "input": {
                "type": "file",
                "file_type": "text",
                "base_dir": "input",
                "file_encoding": "utf-8",
                "file_pattern": ".*\\.txt$"
            },
            "cache": {
                "type": "file",
                "base_dir": "cache"
            },
            "storage": {
                "type": "file",
                "base_dir": "output"
            },
            "reporting": {
                "type": "file",
                "base_dir": "output"
            },
            "entity_extraction": {
                "prompt": "prompts/entity_extraction.txt",
                "entity_types": ["organization", "person", "geo", "event"],
                "max_gleanings": 0
            },
            "summarize_descriptions": {
                "prompt": "prompts/summarize_descriptions.txt",
                "max_length": 500
            },
            "claim_extraction": {
                "prompt": "prompts/claim_extraction.txt",
                "description": "Any claims or facts that could be relevant to information discovery.",
                "max_gleanings": 0
            },
            "community_reports": {
                "prompt": "prompts/community_report.txt",
                "max_length": 2000,
                "max_input_length": 8000
            },
            "group_by_columns": ["id", "short_id"],
            "embed_graph": {
                "enabled": False
            },
            "umap": {
                "enabled": False
            },
            "snapshots": {
                "graphml": False,
                "raw_entities": False,
                "top_level_nodes": False
            },
            "local_search": {
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "top_k_mapped_entities": 10,
                "top_k_relationships": 10,
                "max_tokens": 12000
            },
            "global_search": {
                "max_tokens": 12000,
                "data_max_tokens": 12000,
                "map_max_tokens": 1000,
                "reduce_max_tokens": 2000,
                "concurrency": 32
            }
        }
    
    def create_workspace_config(self, workspace_path: Path, custom_settings: Dict[str, Any] = None) -> bool:
        """Create configuration files for a new workspace."""
        try:
            # Ensure workspace exists
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Copy base configuration if it exists
            if self.base_config_dir.exists():
                self._copy_base_config(workspace_path)
            
            # Create or update settings.yaml
            settings = self.default_settings.copy()
            if custom_settings:
                settings.update(custom_settings)
            
            self._write_settings_yaml(workspace_path, settings)
            
            # Create prompts directory if it doesn't exist
            self._ensure_prompts_directory(workspace_path)
            
            logger.info(f"Created workspace configuration at {workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating workspace config: {e}")
            return False
    
    def _copy_base_config(self, workspace_path: Path):
        """Copy base configuration files to workspace."""
        # Copy settings.yaml if it exists
        base_settings = self.base_config_dir / "settings.yaml"
        if base_settings.exists():
            shutil.copy2(base_settings, workspace_path / "settings.yaml")
        
        # Copy prompts directory if it exists
        base_prompts = self.base_config_dir / "prompts"
        workspace_prompts = workspace_path / "prompts"
        if base_prompts.exists() and base_prompts.is_dir():
            if workspace_prompts.exists():
                shutil.rmtree(workspace_prompts)
            shutil.copytree(base_prompts, workspace_prompts)
    
    def _write_settings_yaml(self, workspace_path: Path, settings: Dict[str, Any]):
        """Write settings.yaml file to workspace."""
        settings_path = workspace_path / "settings.yaml"
        with open(settings_path, 'w', encoding='utf-8') as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
    
    def _ensure_prompts_directory(self, workspace_path: Path):
        """Ensure prompts directory exists with default prompts."""
        prompts_dir = workspace_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        
        # Create default prompts if they don't exist
        default_prompts = {
            "entity_extraction.txt": """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation of why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score between 1 to 10, indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

-Examples-
{examples}

-Real Data-
Entity types: {entity_types}
Text: {input_text}
Output:
""",
            "summarize_descriptions.txt": """
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or more entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
""",
            "community_report.txt": """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations, individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the impact severity of the community within the network (e.g., 0 = low impact, 10 = high impact)
- RATING EXPLANATION: Give a single sentence explanation of the impact severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by an explanation of why the insight is important. Be comprehensive.

Return output as a well-formed JSON object with the following keys:
- title: report title
- summary: executive summary
- rating: impact severity rating (float between 0-10)
- rating_explanation: reasoning for the rating
- findings: list of dict with keys "summary" and "explanation"

# Data

Use the below data to prepare your report. Only use the provided data.

{input_data}

Output:
""",
            "claim_extraction.txt": """
-Goal-
Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims associated with those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the provided claim description, and the entity should be the subject of the claim.
For each claim, extract the following information:
- Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
- Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
- Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
- Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed to be true, FALSE means the claim is confirmed to be false, SUSPECTED means the claim is not verified.
- Claim Description: Detailed description explaining the reasoning behind the claim, together with all the supporting evidence and citations.
- Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
- Claim Source Text: List of all quotes from the original text that are relevant to the claim.

Format each claim as (<|><subject><|><object><|><claim_type><|><claim_status><|><claim_start_date><|><claim_end_date><|><claim_description><|><claim_source_text><|>)

3. Return output in English as a single list of all the claims identified in steps 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

-Examples-
{examples}

-Real Data-
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output:
"""
        }
        
        for prompt_file, content in default_prompts.items():
            prompt_path = prompts_dir / prompt_file
            if not prompt_path.exists():
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(content.strip())
    
    def update_workspace_settings(self, workspace_path: Path, updates: Dict[str, Any]) -> bool:
        """Update settings for an existing workspace."""
        try:
            settings_path = workspace_path / "settings.yaml"
            
            # Load current settings
            if settings_path.exists():
                with open(settings_path, 'r', encoding='utf-8') as f:
                    current_settings = yaml.safe_load(f)
            else:
                current_settings = self.default_settings.copy()
            
            # Apply updates
            current_settings.update(updates)
            
            # Write back
            self._write_settings_yaml(workspace_path, current_settings)
            
            logger.info(f"Updated settings for workspace {workspace_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating workspace settings: {e}")
            return False
    
    def get_workspace_settings(self, workspace_path: Path) -> Dict[str, Any]:
        """Get settings for a workspace."""
        try:
            settings_path = workspace_path / "settings.yaml"
            if settings_path.exists():
                with open(settings_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                return self.default_settings.copy()
        except Exception as e:
            logger.error(f"Error reading workspace settings: {e}")
            return self.default_settings.copy()
    
    def cleanup_workspace(self, workspace_path: Path) -> bool:
        """Clean up workspace files (cache, logs, etc.)."""
        try:
            cleanup_dirs = ["cache", "logs"]
            for cleanup_dir in cleanup_dirs:
                dir_path = workspace_path / cleanup_dir
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned up {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up workspace: {e}")
            return False
    
    def validate_workspace(self, workspace_path: Path) -> Dict[str, bool]:
        """Validate workspace configuration and structure."""
        validation_results = {
            "workspace_exists": workspace_path.exists(),
            "settings_exists": (workspace_path / "settings.yaml").exists(),
            "prompts_dir_exists": (workspace_path / "prompts").exists(),
            "input_dir_exists": (workspace_path / "input").exists(),
            "output_dir_exists": (workspace_path / "output").exists(),
            "cache_dir_exists": (workspace_path / "cache").exists(),
            "api_key_configured": False
        }
        
        # Check API key configuration
        try:
            settings = self.get_workspace_settings(workspace_path)
            api_key = settings.get("llm", {}).get("api_key", "")
            validation_results["api_key_configured"] = bool(api_key and api_key != "${GRAPHRAG_API_KEY}")
        except Exception:
            pass
        
        return validation_results
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get required environment variables for GraphRAG."""
        return {
            "GRAPHRAG_API_KEY": os.getenv("GRAPHRAG_API_KEY", ""),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        }
    
    def check_api_key_availability(self) -> bool:
        """Check if API key is available in environment."""
        env_vars = self.get_environment_variables()
        return any(env_vars.values())

# Global instance
config_manager = ConfigManager()

# Convenience functions
def create_workspace_config(workspace_path: Path, custom_settings: Dict[str, Any] = None) -> bool:
    """Create configuration for a workspace."""
    return config_manager.create_workspace_config(workspace_path, custom_settings)

def update_workspace_settings(workspace_path: Path, updates: Dict[str, Any]) -> bool:
    """Update workspace settings."""
    return config_manager.update_workspace_settings(workspace_path, updates)

def get_workspace_settings(workspace_path: Path) -> Dict[str, Any]:
    """Get workspace settings."""
    return config_manager.get_workspace_settings(workspace_path)

def validate_workspace(workspace_path: Path) -> Dict[str, bool]:
    """Validate workspace configuration."""
    return config_manager.validate_workspace(workspace_path)

def check_api_key_availability() -> bool:
    """Check if API key is available."""
    return config_manager.check_api_key_availability()