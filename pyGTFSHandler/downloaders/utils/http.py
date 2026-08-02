# -*- coding: utf-8 -*-
"""Thin, consistent wrapper around `requests` for downloader API calls.

Each downloader talks to a different REST API, but they all need the same
basic behaviour: send a request with a timeout, raise a clear error if the
HTTP call fails or the response isn't valid JSON, and log enough context to
debug a failure. `request_json` centralizes that so individual downloaders
only need to supply the method, URL, and parameters specific to their API.
"""

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


def request_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """Perform an HTTP request and return its decoded JSON body.

    Args:
        method: HTTP method to use (e.g. "GET", "POST").
        url: Target endpoint URL.
        headers: Optional HTTP headers, such as an API key or Authorization
            header.
        params: Optional query string parameters.
        json: Optional JSON request body, for POST/PUT-style calls.
        timeout: Request timeout in seconds.

    Returns:
        The response body decoded as JSON (list, dict, or scalar).

    Raises:
        requests.exceptions.RequestException: If the request fails at the
            network level or returns a non-2xx status code.
        ValueError: If the response body is not valid JSON.
    """
    try:
        response = requests.request(
            method, url, headers=headers, params=params, json=json, timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        body = getattr(e.response, "text", "")
        logger.error(f"Request {method} {url} failed: {e}. Response body: {body}")
        raise

    try:
        return response.json()
    except ValueError as e:
        logger.error(f"Response from {method} {url} was not valid JSON: {e}")
        raise
