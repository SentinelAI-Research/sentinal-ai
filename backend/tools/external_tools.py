import requests


def call_external_api(
    url: str,
    params: dict | None = None,
    timeout: int = 10,
) -> dict:
    """
    Make an HTTP GET request to an explicitly supplied endpoint.

    SentinelAI will later enforce an external API allowlist
    before this tool is allowed to execute.
    """

    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    response = requests.get(
        url,
        params=params or {},
        timeout=timeout,
    )

    response.raise_for_status()

    try:
        return response.json()
    except ValueError:
        return {
            "status_code": response.status_code,
            "text": response.text,
        }