"""Error handling utilities for langchain-glean package.

This module provides utilities for consistent error handling across
all components of the langchain-glean integration.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from glean.api_client import errors
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun

from langchain_glean.exceptions import (
    GleanAPIError,
    GleanAuthenticationError,
    GleanConfigurationError,
    GleanIntegrationError,
    GleanNotFoundError,
    GleanRateLimitError,
    GleanTimeoutError,
    GleanValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def convert_glean_error(error: Exception, context: str = "") -> GleanIntegrationError:
    """Convert a Glean API error to an appropriate custom exception.
    
    Args:
        error: The original error from Glean API
        context: Additional context about where the error occurred
        
    Returns:
        An appropriate GleanIntegrationError subclass
    """
    if isinstance(error, errors.GleanError):
        message = f"Glean API error{f' in {context}' if context else ''}: {str(error)}"
        
        # Extract additional details from the error if available
        details: Dict[str, Any] = {}
        status_code = None
        error_type = None
        
        if hasattr(error, "raw_response") and error.raw_response:
            details["raw_response"] = error.raw_response
            
        if hasattr(error, "status_code"):
            status_code = error.status_code
            
        if hasattr(error, "error_type"):
            error_type = error.error_type
            
        # Map to specific exception types based on status code or error details
        if status_code == 401 or "authentication" in str(error).lower() or "unauthorized" in str(error).lower():
            return GleanAuthenticationError(message, error, status_code, error_type, details)
        elif status_code == 404 or "not found" in str(error).lower():
            return GleanNotFoundError(message, error, status_code, error_type, details)
        elif status_code == 429 or "rate limit" in str(error).lower() or "too many requests" in str(error).lower():
            return GleanRateLimitError(message, error, status_code, error_type, details)
        elif "timeout" in str(error).lower():
            return GleanTimeoutError(message, error, status_code, error_type, details)
        else:
            return GleanAPIError(message, error, status_code, error_type, details)
    else:
        # For non-Glean errors, wrap them in a generic API error
        message = f"Unexpected error{f' in {context}' if context else ''}: {str(error)}"
        return GleanAPIError(message, error)


def handle_api_error(error: Exception, context: str = "", reraise: bool = True) -> Optional[GleanIntegrationError]:
    """Handle an API error with consistent logging and optional re-raising.
    
    Args:
        error: The error to handle
        context: Context about where the error occurred
        reraise: Whether to re-raise the converted error
        
    Returns:
        The converted error if reraise=False, None otherwise
        
    Raises:
        GleanIntegrationError: The converted error if reraise=True
    """
    converted_error = convert_glean_error(error, context)
    
    # Log the error with appropriate level
    if isinstance(converted_error, (GleanAuthenticationError, GleanConfigurationError)):
        logger.error("Configuration/Authentication error: %s", converted_error)
    elif isinstance(converted_error, GleanRateLimitError):
        logger.warning("Rate limit error: %s", converted_error)
    elif isinstance(converted_error, GleanNotFoundError):
        logger.info("Resource not found: %s", converted_error)
    else:
        logger.error("API error: %s", converted_error)
    
    if reraise:
        raise converted_error
    return converted_error


async def handle_retriever_error(
    error: Exception,
    run_manager: AsyncCallbackManagerForRetrieverRun,
    context: str = "",
    reraise: bool = False,
) -> None:
    """Handle retriever errors with async callback manager.
    
    Args:
        error: The error to handle
        run_manager: The async callback manager
        context: Context about where the error occurred
        reraise: Whether to re-raise the error after handling
    """
    converted_error = convert_glean_error(error, context)
    await run_manager.on_retriever_error(converted_error)
    
    if reraise:
        raise converted_error


def handle_retriever_error_sync(
    error: Exception,
    run_manager: CallbackManagerForRetrieverRun,
    context: str = "",
    reraise: bool = False,
) -> None:
    """Handle retriever errors with sync callback manager.
    
    Args:
        error: The error to handle
        run_manager: The sync callback manager
        context: Context about where the error occurred
        reraise: Whether to re-raise the error after handling
    """
    converted_error = convert_glean_error(error, context)
    run_manager.on_retriever_error(converted_error)
    
    if reraise:
        raise converted_error


def validate_configuration(instance: str, api_token: str, act_as: Optional[str] = None) -> None:
    """Validate Glean configuration parameters.
    
    Args:
        instance: Glean instance/subdomain
        api_token: Glean API token
        act_as: Optional email to act as
        
    Raises:
        GleanConfigurationError: If configuration is invalid
    """
    if not instance or not instance.strip():
        raise GleanConfigurationError("Glean instance is required")
        
    if not api_token or not api_token.strip():
        raise GleanConfigurationError("Glean API token is required")
        
    # Basic validation of instance format (should be alphanumeric, possibly with hyphens)
    if not instance.replace("-", "").replace("_", "").isalnum():
        raise GleanConfigurationError(f"Invalid Glean instance format: {instance}")
        
    # Basic validation of API token format (should not be empty or whitespace-only)
    if len(api_token.strip()) < 10:  # Reasonable minimum length for API tokens
        raise GleanConfigurationError("API token appears to be invalid (too short)")


def safe_execute(func, *args, default_return=None, context: str = "", reraise_on: Optional[Type[Exception]] = None, **kwargs):
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        default_return: Value to return if function fails
        context: Context for error logging
        reraise_on: Exception type to always re-raise
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if reraise_on and isinstance(e, reraise_on):
            raise
            
        handle_api_error(e, context, reraise=False)
        return default_return