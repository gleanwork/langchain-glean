from typing import Dict, Optional


class GleanAuth:
    """
    Authentication settings for Glean API.
    """

    api_token: str
    subdomain: str
    act_as: Optional[str] = None
    is_oauth: Optional[bool] = False

    def __init__(
        self,
        api_token: str,
        subdomain: str,
        act_as: Optional[str] = None,
        is_oauth: Optional[bool] = False,
    ):
        """
        Initialize GleanAuth instance.

        Args:
            api_token: API token for authenticating with Glean
            subdomain: Subdomain for Glean API
            is_oauth: Optional flag to indicate to use OAuth headers
            act_as: Optional user to act as when authenticating with Glean
        """

        self.api_token = api_token
        self.subdomain = subdomain
        self.is_oauth = is_oauth
        self.act_as = act_as

    def get_base_url(self, path: str = "rest/api", version: str = "v1") -> str:
        """
        Return the base URL for the Glean API request.

        Args:
            path: Optional path to append to the base URL
            version: Optional version to append to the base URL

        Returns:
            str: Base URL for the Glean API request
        """

        base_url = f"https://{self.subdomain}-be.glean.com"

        if path:
            base_url += "/" + path

        if version:
            base_url += "/" + version

        return base_url

    def get_headers(self) -> Dict[str, str]:
        """
        Return the auth headers for the Glean API request.

        Returns:
            Dict[str, str]: Headers for the Glean API request
        """

        # set the headers based on the auth type
        headers = {}

        # https://developers.glean.com/docs/client_api/client_api_scopes/#using-access-tokens
        headers["Authorization"] = f"Bearer {self.api_token}"

        # https://developers.glean.com/docs/client_api/client_api_scopes/#using-access-tokens
        if self.is_oauth:
            headers["X-Glean-Auth-Type"] = "OAUTH"

        # https://developers.glean.com/docs/client_api/client_api_scopes/#users
        if self.act_as:
            headers["X-Scio-Actas"] = self.act_as

        return headers
