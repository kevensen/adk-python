"""
Tests for MCP Server Sampling functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any, Dict, Optional

from google.adk.tools.mcp_tool.sampling_config import SamplingConfig
from google.adk.tools.mcp_tool.mcp_sampling_handler import MCPSamplingHandler
from google.adk.tools.mcp_tool.model_selection_engine import ModelSelectionEngine, ModelMetadata
from google.adk.tools.mcp_tool.sampling_approval_manager import SamplingApprovalManager


class TestSamplingConfig:
    """Test cases for SamplingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SamplingConfig()
        
        assert config.enable_sampling is False
        assert config.require_user_approval is True
        assert config.default_model is None
        assert config.allowed_models is None
        assert config.max_tokens_per_request == 4096
        assert config.rate_limit_per_minute == 10
        assert config.auto_approve_trusted_servers is False
    
    def test_enabled_config(self):
        """Test enabled sampling configuration."""
        config = SamplingConfig(
            enable_sampling=True,
            require_user_approval=False,
            default_model="gemini-2.0-flash",
            allowed_models=["gemini-2.0-flash", "gemini-1.5-pro"],
            max_tokens_per_request=2048,
            rate_limit_per_minute=5
        )
        
        assert config.enable_sampling is True
        assert config.require_user_approval is False
        assert config.default_model == "gemini-2.0-flash"
        assert config.allowed_models == ["gemini-2.0-flash", "gemini-1.5-pro"]
        assert config.max_tokens_per_request == 2048
        assert config.rate_limit_per_minute == 5


class TestModelSelectionEngine:
    """Test cases for ModelSelectionEngine."""
    
    def test_select_model_with_default(self):
        """Test model selection with default model."""
        config = SamplingConfig(
            enable_sampling=True,
            default_model="gemini-2.0-flash"
        )
        engine = ModelSelectionEngine(config)
        
        # No preferences provided, should use default
        model = engine.select_model(None)
        assert model == "gemini-2.0-flash"
    
    def test_select_model_with_preferences(self):
        """Test model selection with model preferences."""
        config = SamplingConfig(
            enable_sampling=True,
            allowed_models=["gemini-2.0-flash", "gemini-1.5-pro", "claude-3-sonnet"]
        )
        engine = ModelSelectionEngine(config)
        
        # Mock ModelPreferences
        preferences = Mock()
        preferences.hints = ["speed", "coding"]
        preferences.cost_priority = 0.7
        preferences.speed_priority = 0.9
        preferences.intelligence_priority = 0.5
        
        model = engine.select_model(preferences)
        # Should select based on speed priority and coding hint
        assert model in config.allowed_models
    
    def test_select_model_restricted(self):
        """Test model selection with restrictions."""
        config = SamplingConfig(
            enable_sampling=True,
            allowed_models=["gemini-1.5-pro"],
            default_model="gemini-2.0-flash"  # Not in allowed_models
        )
        engine = ModelSelectionEngine(config)
        
        model = engine.select_model(None)
        # Should use allowed model instead of default
        assert model == "gemini-1.5-pro"


class TestSamplingApprovalManager:
    """Test cases for SamplingApprovalManager."""
    
    @pytest.mark.asyncio
    async def test_approval_not_required(self):
        """Test when approval is not required."""
        config = SamplingConfig(
            enable_sampling=True,
            require_user_approval=False
        )
        manager = SamplingApprovalManager(config)
        
        # Mock sampling request
        request = Mock()
        request.arguments = {"prompt": {"type": "text", "text": "Test prompt"}}
        request.preferences = None
        
        approved = await manager.request_approval(request, "gemini-2.0-flash")
        assert approved is True
    
    @pytest.mark.asyncio
    async def test_auto_approve_trusted(self):
        """Test auto-approval for trusted servers."""
        config = SamplingConfig(
            enable_sampling=True,
            require_user_approval=True,
            auto_approve_trusted_servers=True
        )
        manager = SamplingApprovalManager(config)
        
        request = Mock()
        request.arguments = {"prompt": {"type": "text", "text": "Test prompt"}}
        request.preferences = None
        
        # Mock as trusted server
        with patch.object(manager, '_is_trusted_server', return_value=True):
            approved = await manager.request_approval(request, "gemini-2.0-flash")
            assert approved is True


class TestMCPSamplingHandler:
    """Test cases for MCPSamplingHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SamplingConfig(
            enable_sampling=True,
            require_user_approval=False,
            default_model="gemini-2.0-flash",
            max_tokens_per_request=2048
        )
        
        # Mock dependencies
        self.mock_registry = Mock()
        self.mock_execute_llm = AsyncMock()
        
        self.handler = MCPSamplingHandler(
            config=self.config,
            llm_registry=self.mock_registry,
            execute_llm_func=self.mock_execute_llm
        )
    
    @pytest.mark.asyncio
    async def test_handle_sampling_request_success(self):
        """Test successful sampling request handling."""
        # Mock sampling request
        request = Mock()
        request.arguments = {
            "prompt": {"type": "text", "text": "Generate a Python function"},
            "max_tokens": 500
        }
        request.preferences = None
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "def example_function():\n    pass"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 150
        self.mock_execute_llm.return_value = mock_response
        
        # Execute request
        result = await self.handler.handle_sampling_request(request)
        
        # Verify result
        assert result is not None
        assert "role" in result
        assert "content" in result
        assert result["role"] == "assistant"
        
        # Verify LLM was called
        self.mock_execute_llm.assert_called_once()
        call_args = self.mock_execute_llm.call_args[1]
        assert call_args["request"].model == "gemini-2.0-flash"
        assert call_args["request"].max_output_tokens == 500
    
    @pytest.mark.asyncio
    async def test_handle_sampling_request_token_limit_exceeded(self):
        """Test handling when token limit is exceeded."""
        request = Mock()
        request.arguments = {
            "prompt": {"type": "text", "text": "Generate a Python function"},
            "max_tokens": 5000  # Exceeds config limit of 2048
        }
        request.preferences = None
        
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            await self.handler.handle_sampling_request(request)
    
    @pytest.mark.asyncio
    async def test_handle_sampling_request_approval_denied(self):
        """Test handling when user approval is denied."""
        # Configure to require approval
        self.handler.config.require_user_approval = True
        
        request = Mock()
        request.arguments = {
            "prompt": {"type": "text", "text": "Generate a Python function"}
        }
        request.preferences = None
        
        # Mock approval as denied
        with patch.object(self.handler.approval_manager, 'request_approval', 
                         return_value=asyncio.create_task(asyncio.coroutine(lambda: False)())):
            with pytest.raises(ValueError, match="User denied"):
                await self.handler.handle_sampling_request(request)
    
    @pytest.mark.asyncio
    async def test_convert_mcp_prompt_to_llm_request(self):
        """Test MCP prompt conversion to LLM request."""
        mcp_prompt = {
            "type": "text",
            "text": "Explain quantum computing"
        }
        
        llm_request = await self.handler._convert_mcp_prompt_to_llm_request(
            mcp_prompt, "gemini-1.5-pro", 1000
        )
        
        assert llm_request.model == "gemini-1.5-pro"
        assert llm_request.max_output_tokens == 1000
        assert len(llm_request.contents) == 1
        assert llm_request.contents[0].text == "Explain quantum computing"
    
    @pytest.mark.asyncio
    async def test_sampling_disabled(self):
        """Test behavior when sampling is disabled."""
        disabled_config = SamplingConfig(enable_sampling=False)
        disabled_handler = MCPSamplingHandler(
            config=disabled_config,
            llm_registry=self.mock_registry,
            execute_llm_func=self.mock_execute_llm
        )
        
        request = Mock()
        request.arguments = {"prompt": {"type": "text", "text": "Test"}}
        
        with pytest.raises(ValueError, match="Sampling is not enabled"):
            await disabled_handler.handle_sampling_request(request)


class TestMCPToolsetIntegration:
    """Integration tests for MCPToolset with sampling."""
    
    def test_toolset_with_sampling_config(self):
        """Test MCPToolset accepts sampling configuration."""
        from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams
        
        sampling_config = SamplingConfig(
            enable_sampling=True,
            default_model="gemini-2.0-flash"
        )
        
        connection_params = StdioConnectionParams(
            command="echo",
            args=["test"],
            timeout=5.0
        )
        
        # Should not raise any errors
        toolset = MCPToolset(
            connection_params=connection_params,
            sampling_config=sampling_config
        )
        
        assert toolset._sampling_config == sampling_config
    
    def test_toolset_without_sampling_config(self):
        """Test MCPToolset works without sampling configuration."""
        from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams
        
        connection_params = StdioConnectionParams(
            command="echo",
            args=["test"],
            timeout=5.0
        )
        
        # Should work with default (disabled) sampling
        toolset = MCPToolset(connection_params=connection_params)
        
        # Should have None or default config
        assert toolset._sampling_config is None or not toolset._sampling_config.enable_sampling


if __name__ == "__main__":
    pytest.main([__file__])
