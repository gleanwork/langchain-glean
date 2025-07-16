"""Custom exceptions for langchain-glean package.

This module defines exception classes that provide structured error handling
for various failure scenarios when interacting with the Glean API.
"""

from typing import Any, Dict, Optional


class GleanIntegrationError(Exception):
    """Base exception for all langchain-glean integration errors.
    
    This is the base class for all exceptions raised by this package.
    Users can catch this to handle all integration-specific errors.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.details = details or {}


class GleanAPIError(GleanIntegrationError):
    """Exception raised when Glean API calls fail.
    
    This exception wraps underlying Glean API errors and provides
    additional context about the failure.
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the API error.
        
        Args:
            message: Human-readable error message
            original_error: The original exception that caused this error
            status_code: HTTP status code if available
            error_type: Type of error from Glean API
            details: Additional error details
        """
        super().__init__(message, details)
        self.original_error = original_error
        self.status_code = status_code
        self.error_type = error_type

    def __str__(self) -> str:
        """String representation with additional context."""
        parts = [super().__str__()]
        
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        
        if self.error_type:
            parts.append(f"Type: {self.error_type}")
            
        if self.original_error:
            parts.append(f"Original: {str(self.original_error)}")
            
        return " | ".join(parts)


class GleanConfigurationError(GleanIntegrationError):
    """Exception raised when configuration is invalid or missing.
    
    This includes missing API tokens, invalid instance names, etc.
    """

    pass


class GleanValidationError(GleanIntegrationError):
    """Exception raised when input validation fails.
    
    This includes invalid query parameters, malformed requests, etc.
    """

    pass


class GleanTimeoutError(GleanAPIError):
    """Exception raised when API calls timeout."""

    pass


class GleanAuthenticationError(GleanAPIError):
    """Exception raised when authentication fails."""

    pass


class GleanNotFoundError(GleanAPIError):
    """Exception raised when requested resources are not found."""

    pass


class GleanRateLimitError(GleanAPIError):
    """Exception raised when rate limits are exceeded."""

    pass