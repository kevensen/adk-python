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

from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class SamplingConfig(BaseModel):
  """Configuration for MCP sampling functionality."""
  
  enable_sampling: bool = False
  """Whether to enable MCP sampling support."""
  
  require_user_approval: bool = True
  """Whether to require user approval for all sampling requests."""
  
  default_model: Optional[str] = None
  """Default model to use if no preference is specified."""
  
  max_tokens_per_request: int = 4096
  """Maximum number of tokens allowed per sampling request."""
  
  allowed_models: Optional[List[str]] = None
  """List of models allowed for sampling. If None, all available models are allowed."""
  
  rate_limit_per_minute: int = 10
  """Maximum number of sampling requests allowed per minute."""
  
  auto_approve_trusted_servers: bool = False
  """Whether to automatically approve sampling requests from trusted servers."""
