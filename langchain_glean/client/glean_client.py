from json import JSONDecodeError
from typing import Any, Dict, Literal, Optional, Union, cast

import requests
from requests.exceptions import HTTPError, RequestException

from langchain_glean.client.glean_auth import GleanAuth

DEFAULT_TIMEOUT = 60


class GleanClientError(Exception):
    """Base exception for GleanClient errors."""

    pass


class GleanHTTPError(GleanClientError):
    """Exception raised for HTTP errors from the Glean API."""

    def __init__(self, status_code: int, message: str, response: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.response = response
        super().__init__(f"HTTP Error {status_code}: {message}")


class GleanConnectionError(GleanClientError):
    """Exception raised for connection errors when communicating with the Glean API."""

    pass


class GleanClient:
    """
    Client for interacting with Glean's REST API.

    This class provides a simple interface for making authenticated requests to Glean's API endpoints.

    Args:
        subdomain: Subdomain for Glean API
        api_token: API token for authenticating with Glean
        act_as: Optional user to act as when authenticating with Glean
        auth_type: Optional authentication type for Glean API
        api_set: Optional API set to use
    """

    base_url: str
    session: requests.Session

    def __init__(
        self,
        auth: GleanAuth,
        *,
        timeout: Optional[int] = DEFAULT_TIMEOUT,
        api_root: Literal[None, "index", "client"] = None,
    ):
        self.session = requests.Session()
        self._timeout = timeout
        self.session.headers.update(cast(Dict[str, str], auth.get_headers()))

        if api_root is None or api_root == "client":
            self.base_url = auth.get_base_url("rest/api")
        else:
            self.base_url = auth.get_base_url("api/index")

    def parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse the response from the Glean API.

        Args:
            response: Response from the Glean API

        Returns:
            Dict containing the API response

        Raises:
            GleanHTTPError: If the response contains an HTTP error
        """
        try:
            response.raise_for_status()
        except HTTPError as e:
            error_response = None
            try:
                error_response = response.json()
            except JSONDecodeError:
                pass

            raise GleanHTTPError(status_code=response.status_code, message=str(e), response=error_response) from e

        body: Union[Dict[str, Any], str] = {}

        try:
            body = response.json()
        except JSONDecodeError:
            body = response.text

        if isinstance(body, str):
            return {"text": body}
        return body

    def post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """
        Send a POST request to the Glean API.

        Args:
            endpoint: API endpoint to call
            **kwargs: Additional arguments to pass to the request

        Returns:
            Dict containing the API response

        Raises:
            GleanHTTPError: If the response contains an HTTP error
            GleanConnectionError: If there's a connection error
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.post(url, **kwargs)
            return response
        except RequestException as e:
            if isinstance(e, HTTPError):
                # This is already handled in parse_response
                raise
            raise GleanConnectionError(f"Connection error when calling {endpoint}: {str(e)}") from e

    def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """
        Send a GET request to the Glean API.

        Args:
            endpoint: API endpoint to call
            **kwargs: Additional arguments to pass to the request

        Returns:
            Dict containing the API response

        Raises:
            GleanHTTPError: If the response contains an HTTP error
            GleanConnectionError: If there's a connection error
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, **kwargs)
            return response
        except RequestException as e:
            if isinstance(e, HTTPError):
                # This is already handled in parse_response
                raise
            raise GleanConnectionError(f"Connection error when calling {endpoint}: {str(e)}") from e
