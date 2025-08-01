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

import asyncio
import logging
import time
from typing import Dict
from typing import List
from typing import Optional

from google.genai import types

try:
  from mcp.types import CreateMessageRequestParams
  from mcp.types import CreateMessageResult
  from mcp.types import ModelPreferences
  from mcp.types import Role
  from mcp.types import SamplingMessage
  from mcp.types import StopReason
  from mcp.types import TextContent
except ImportError as e:
  import sys
  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...models.registry import LLMRegistry
from .model_selection_engine import ModelSelectionEngine
from .sampling_approval_manager import SamplingApprovalManager
from .sampling_config import SamplingConfig

logger = logging.getLogger('google_adk.' + __name__)


class MCPSamplingHandler:
  """Handles MCP sampling requests and bridges to ADK LLM infrastructure."""
  
  def __init__(self, config: SamplingConfig):
    """Initialize the sampling handler.
    
    Args:
      config: Configuration for sampling behavior.
    """
    self.config = config
    self.model_selector = ModelSelectionEngine(config)
    self.approval_manager = SamplingApprovalManager(config)
    self._request_times: List[float] = []
    
  async def handle_sampling_request(
      self, 
      request: CreateMessageRequestParams
  ) -> CreateMessageResult:
    """Handle a sampling/createMessage request from MCP server.
    
    Args:
      request: The MCP sampling request parameters.
      
    Returns:
      The result of the sampling request.
      
    Raises:
      ValueError: If the request is invalid or user rejects the request.
      RuntimeError: If rate limits are exceeded.
    """
    logger.info('Received MCP sampling request with %d messages', len(request.messages))
    
    # 1. Validate request
    self._validate_request(request)
    
    # 2. Check rate limits
    self._check_rate_limits()
    
    # 3. Select appropriate model
    selected_model = self.model_selector.select_model(request.modelPreferences)
    logger.info('Selected model: %s', selected_model)
    
    # 4. Request user approval if required
    if self.config.require_user_approval and not self.config.auto_approve_trusted_servers:
      approved = await self.approval_manager.request_approval(request, selected_model)
      if not approved:
        logger.warning('User rejected sampling request')
        raise ValueError('User rejected sampling request')
    
    # 5. Convert to ADK LlmRequest
    llm_request = self._convert_to_llm_request(request, selected_model)
    
    # 6. Execute via ADK LLM infrastructure
    llm_response = await self._execute_llm_request(llm_request)
    
    # 7. Convert back to MCP format
    result = self._convert_to_mcp_result(llm_response, selected_model)
    
    logger.info('Successfully completed sampling request')
    return result
  
  def _validate_request(self, request: CreateMessageRequestParams) -> None:
    """Validate the sampling request parameters.
    
    Args:
      request: The request to validate.
      
    Raises:
      ValueError: If the request is invalid.
    """
    if not request.messages:
      raise ValueError('Sampling request must contain at least one message')
    
    if request.maxTokens > self.config.max_tokens_per_request:
      raise ValueError(
          f'Requested token count ({request.maxTokens}) exceeds maximum allowed '
          f'({self.config.max_tokens_per_request})'
      )
    
    if request.maxTokens <= 0:
      raise ValueError('Max tokens must be positive')
    
    # Validate message content
    for message in request.messages:
      if not isinstance(message.content, (TextContent,)):
        raise ValueError('Only text content is currently supported for sampling')
  
  def _check_rate_limits(self) -> None:
    """Check if rate limits are exceeded.
    
    Raises:
      RuntimeError: If rate limits are exceeded.
    """
    current_time = time.time()
    
    # Remove requests older than 1 minute
    self._request_times = [
        t for t in self._request_times 
        if current_time - t < 60
    ]
    
    if len(self._request_times) >= self.config.rate_limit_per_minute:
      raise RuntimeError(
          f'Rate limit exceeded: {self.config.rate_limit_per_minute} requests per minute'
      )
    
    self._request_times.append(current_time)
  
  def _convert_to_llm_request(
      self, 
      request: CreateMessageRequestParams, 
      selected_model: str
  ) -> LlmRequest:
    """Convert MCP request to ADK LlmRequest.
    
    Args:
      request: The MCP sampling request.
      selected_model: The selected model identifier.
      
    Returns:
      An ADK LlmRequest object.
    """
    # Convert MCP messages to ADK Content format
    contents = []
    
    # Add system prompt if provided
    if request.systemPrompt:
      contents.append(types.Content(
          role='user',
          parts=[types.Part.from_text(request.systemPrompt)]
      ))
    
    # Convert MCP messages
    for mcp_message in request.messages:
      if isinstance(mcp_message.content, TextContent):
        role = 'user' if mcp_message.role == Role.user else 'model'
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(mcp_message.content.text)]
        ))
    
    # Create generation config
    config = types.GenerateContentConfig()
    if request.maxTokens:
      config.max_output_tokens = request.maxTokens
    if request.temperature is not None:
      config.temperature = request.temperature
    if request.stopSequences:
      config.stop_sequences = request.stopSequences
    
    return LlmRequest(
        model=selected_model,
        contents=contents,
        config=config
    )
  
  async def _execute_llm_request(self, llm_request: LlmRequest) -> LlmResponse:
    """Execute the LLM request using ADK infrastructure.
    
    Args:
      llm_request: The ADK LLM request to execute.
      
    Returns:
      The LLM response.
      
    Raises:
      RuntimeError: If the LLM call fails.
    """
    try:
      # Get the LLM instance from registry
      llm = LLMRegistry.new_llm(llm_request.model)
      
      # Execute the request
      response_generator = llm.generate_content_async(llm_request, stream=False)
      response = await response_generator.__anext__()
      
      return response
    except Exception as e:
      logger.error('Failed to execute LLM request: %s', str(e))
      raise RuntimeError(f'LLM execution failed: {str(e)}') from e
  
  def _convert_to_mcp_result(
      self, 
      llm_response: LlmResponse, 
      model_name: str
  ) -> CreateMessageResult:
    """Convert ADK LlmResponse to MCP CreateMessageResult.
    
    Args:
      llm_response: The ADK LLM response.
      model_name: The name of the model that generated the response.
      
    Returns:
      An MCP CreateMessageResult object.
    """
    # Extract text content from response
    text_content = ''
    if llm_response.content and llm_response.content.parts:
      text_parts = [
          part.text for part in llm_response.content.parts 
          if part.text
      ]
      text_content = ''.join(text_parts)
    
    # Determine stop reason
    stop_reason = None
    if hasattr(llm_response, 'finish_reason'):
      if llm_response.finish_reason == 'STOP':
        stop_reason = StopReason.endTurn
      elif llm_response.finish_reason == 'MAX_TOKENS':
        stop_reason = StopReason.maxTokens
    
    return CreateMessageResult(
        role=Role.assistant,
        content=TextContent(type='text', text=text_content),
        model=model_name,
        stopReason=stop_reason
    )
