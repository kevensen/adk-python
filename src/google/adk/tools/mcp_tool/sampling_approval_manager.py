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
from typing import Optional

try:
  from mcp.types import CreateMessageRequestParams
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

from .sampling_config import SamplingConfig

logger = logging.getLogger('google_adk.' + __name__)


class SamplingApprovalManager:
  """Manages user approval workflow for MCP sampling requests."""
  
  def __init__(self, config: SamplingConfig):
    """Initialize the approval manager.
    
    Args:
      config: Configuration for sampling behavior.
    """
    self.config = config
  
  async def request_approval(
      self, 
      request: CreateMessageRequestParams, 
      selected_model: str
  ) -> bool:
    """Request user approval for a sampling request.
    
    Args:
      request: The MCP sampling request to approve.
      selected_model: The selected model for the request.
      
    Returns:
      True if approved, False if rejected.
    """
    # For now, implement a simple console-based approval
    # In a real implementation, this would integrate with the UI framework
    # to show a proper approval dialog
    
    print("\n" + "="*60)
    print("MCP SAMPLING REQUEST - USER APPROVAL REQUIRED")
    print("="*60)
    print(f"Model: {selected_model}")
    print(f"Max Tokens: {request.maxTokens}")
    
    if request.systemPrompt:
      print(f"System Prompt: {request.systemPrompt}")
    
    print(f"Messages ({len(request.messages)}):")
    for i, message in enumerate(request.messages):
      if isinstance(message.content, TextContent):
        content_preview = message.content.text[:200]
        if len(message.content.text) > 200:
          content_preview += "..."
        print(f"  {i+1}. [{message.role}] {content_preview}")
    
    if request.temperature is not None:
      print(f"Temperature: {request.temperature}")
    
    if request.stopSequences:
      print(f"Stop Sequences: {request.stopSequences}")
    
    print("\nDo you want to approve this sampling request?")
    
    # In a production environment, this would be replaced with proper UI integration
    try:
      response = input("Enter 'y' to approve, 'n' to reject: ").strip().lower()
      approved = response in ['y', 'yes', '1', 'true']
      
      if approved:
        print("✓ Request approved")
      else:
        print("✗ Request rejected")
      
      print("="*60)
      return approved
      
    except (EOFError, KeyboardInterrupt):
      print("\n✗ Request rejected (interrupted)")
      print("="*60)
      return False
  
  def _sanitize_content_for_display(self, text: str, max_length: int = 500) -> str:
    """Sanitize and truncate content for safe display.
    
    Args:
      text: The text content to sanitize.
      max_length: Maximum length to display.
      
    Returns:
      Sanitized and truncated text.
    """
    # Remove potentially sensitive patterns (basic implementation)
    sanitized = text.replace('\x00', '').replace('\x1b', '')
    
    if len(sanitized) > max_length:
      sanitized = sanitized[:max_length] + "..."
    
    return sanitized
