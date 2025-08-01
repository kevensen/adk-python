# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .sampling_config import SamplingConfig

try:
  from mcp.types import ModelHint
  from mcp.types import ModelPreferences
except ImportError as e:
  import sys
  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

from ...models.registry import LLMRegistry

logger = logging.getLogger('google_adk.' + __name__)


class ModelMetadata:
  """Metadata about a model's capabilities and characteristics."""
  
  def __init__(
      self,
      name: str,
      cost_score: float = 0.5,
      speed_score: float = 0.5,
      intelligence_score: float = 0.5,
      aliases: Optional[List[str]] = None
  ):
    """Initialize model metadata.
    
    Args:
      name: The model name.
      cost_score: Cost efficiency score (0.0 = expensive, 1.0 = cheap).
      speed_score: Speed score (0.0 = slow, 1.0 = fast).
      intelligence_score: Intelligence/capability score (0.0 = basic, 1.0 = advanced).
      aliases: Alternative names that might match this model.
    """
    self.name = name
    self.cost_score = cost_score
    self.speed_score = speed_score
    self.intelligence_score = intelligence_score
    self.aliases = aliases or []


class ModelSelectionEngine:
  """Selects appropriate ADK model based on MCP preferences."""
  
  def __init__(self, config: 'SamplingConfig'):
    """Initialize the model selection engine.
    
    Args:
      config: The sampling configuration containing model restrictions.
    """
    self.config = config
    self.model_metadata = self._load_model_metadata()
    
  def _load_model_metadata(self) -> Dict[str, ModelMetadata]:
    """Load metadata for available models.
    
    Returns:
      Dictionary mapping model names to their metadata.
    """
    # Define metadata for common models
    # These scores are approximate and can be adjusted based on actual characteristics
    metadata = {
        'gemini-2.0-flash': ModelMetadata(
            name='gemini-2.0-flash',
            cost_score=0.7,  # Relatively cost-effective
            speed_score=0.9,  # Fast
            intelligence_score=0.8,  # High capability
            aliases=['gemini-2-flash', 'gemini-2.0', 'gemini-flash']
        ),
        'gemini-1.5-pro': ModelMetadata(
            name='gemini-1.5-pro',
            cost_score=0.4,  # More expensive
            speed_score=0.6,  # Moderate speed
            intelligence_score=0.9,  # Very high capability
            aliases=['gemini-1.5', 'gemini-pro', 'gemini-1-5-pro']
        ),
        'gemini-1.5-flash': ModelMetadata(
            name='gemini-1.5-flash',
            cost_score=0.8,  # Cost-effective
            speed_score=0.9,  # Fast
            intelligence_score=0.7,  # Good capability
            aliases=['gemini-1.5', 'gemini-flash', 'gemini-1-5-flash']
        ),
        # Add metadata for other models as needed
        'claude-3-sonnet': ModelMetadata(
            name='claude-3-sonnet',
            cost_score=0.5,
            speed_score=0.7,
            intelligence_score=0.9,
            aliases=['claude-3', 'claude-sonnet', 'sonnet']
        ),
        'gpt-4': ModelMetadata(
            name='gpt-4',
            cost_score=0.3,
            speed_score=0.5,
            intelligence_score=0.9,
            aliases=['gpt4', 'openai-gpt-4']
        ),
        'gpt-3.5-turbo': ModelMetadata(
            name='gpt-3.5-turbo',
            cost_score=0.8,
            speed_score=0.8,
            intelligence_score=0.7,
            aliases=['gpt-3.5', 'gpt35', 'openai-gpt-3.5']
        ),
    }
    
    return metadata
  
  def select_model(
      self, 
      preferences: Optional[ModelPreferences] = None
  ) -> str:
    """Select best model based on preferences and availability.
    
    Args:
      preferences: MCP model preferences from the sampling request.
      
    Returns:
      The name of the selected model.
      
    Raises:
      ValueError: If no suitable model is found.
    """
    available_models = self._get_available_models(self.config.allowed_models)
    
    if not available_models:
      raise ValueError('No available models found')
    
    if not preferences:
      return self._get_default_model(available_models)
    
    # Try model hints first
    if preferences.hints:
      for hint in preferences.hints:
        matched_model = self._match_hint_to_model(hint, available_models)
        if matched_model:
          logger.info('Selected model based on hint "%s": %s', hint.name, matched_model)
          return matched_model
    
    # Fall back to priority-based selection
    selected_model = self._select_by_priorities(preferences, available_models)
    logger.info('Selected model based on priorities: %s', selected_model)
    return selected_model
  
  def _get_available_models(self, allowed_models: Optional[List[str]]) -> List[str]:
    """Get list of available models.
    
    Args:
      allowed_models: Optional list of allowed models.
      
    Returns:
      List of available model names.
    """
    # Get all supported models from the registry
    all_models = list(self.model_metadata.keys())
    
    # Filter by allowed models if specified
    if allowed_models:
      available = [model for model in all_models if model in allowed_models]
    else:
      available = all_models
    
    return available
  
  def _get_default_model(self, available_models: List[str]) -> str:
    """Get the default model from available models.
    
    Args:
      available_models: List of available model names.
      
    Returns:
      The default model name.
    """
    # Use configured default model if available
    if self.config.default_model and self.config.default_model in available_models:
      return self.config.default_model
    
    # Prefer Gemini 2.0 Flash as default if available
    preferred_defaults = [
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    
    for model in preferred_defaults:
      if model in available_models:
        return model
    
    # Return first available model if no preferred default is found
    return available_models[0]
  
  def _match_hint_to_model(self, hint: ModelHint, available_models: List[str]) -> Optional[str]:
    """Match a model hint to an available model.
    
    Args:
      hint: The model hint to match.
      available_models: List of available model names.
      
    Returns:
      The matched model name, or None if no match is found.
    """
    hint_name = hint.name.lower()
    
    # Direct name match
    for model in available_models:
      if hint_name in model.lower():
        return model
    
    # Check aliases
    for model in available_models:
      if model in self.model_metadata:
        metadata = self.model_metadata[model]
        for alias in metadata.aliases:
          if hint_name in alias.lower() or alias.lower() in hint_name:
            return model
    
    return None
  
  def _select_by_priorities(
      self, 
      preferences: ModelPreferences, 
      available_models: List[str]
  ) -> str:
    """Select model based on priority scores.
    
    Args:
      preferences: The model preferences with priority scores.
      available_models: List of available model names.
      
    Returns:
      The selected model name.
    """
    best_model = None
    best_score = -1.0
    
    for model in available_models:
      if model not in self.model_metadata:
        continue
        
      metadata = self.model_metadata[model]
      
      # Calculate weighted score based on preferences
      score = 0.0
      total_weight = 0.0
      
      if preferences.costPriority is not None:
        score += preferences.costPriority * metadata.cost_score
        total_weight += preferences.costPriority
        
      if preferences.speedPriority is not None:
        score += preferences.speedPriority * metadata.speed_score
        total_weight += preferences.speedPriority
        
      if preferences.intelligencePriority is not None:
        score += preferences.intelligencePriority * metadata.intelligence_score
        total_weight += preferences.intelligencePriority
      
      # Normalize score
      if total_weight > 0:
        score = score / total_weight
      else:
        # If no priorities specified, use balanced score
        score = (metadata.cost_score + metadata.speed_score + metadata.intelligence_score) / 3.0
      
      if score > best_score:
        best_score = score
        best_model = model
    
    if best_model is None:
      # Fallback to first available model
      best_model = available_models[0]
    
    return best_model
