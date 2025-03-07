from json import JSONDecodeError
from typing import Any, Dict, Literal, Optional

import requests

from .glean_auth import GleanAuth

DEFAULT_TIMEOUT = 60


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
        self.session.timeout = timeout
        self.session.headers = auth.get_headers()

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
        """
        response.raise_for_status()

        body = None

        try:
            body = response.json()
        except JSONDecodeError:
            body = response.text

        return body

    def post(self, endpoint, **kwargs):
        """
        Send a POST request to the Glean API.

        Args:
            endpoint: API endpoint to call
            **kwargs: Additional arguments to pass to the request

        Returns:
            Dict containing the API response
        """
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(url, **kwargs)

        return self.parse_response(response)

    def get(self, endpoint, **kwargs):
        """
        Send a GET request to the Glean API.

        Args:
            endpoint: API endpoint to call
            **kwargs: Additional arguments to pass to the request

        Returns:
            Dict containing the API response
        """

        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, **kwargs)

        return self.parse_response(response)
