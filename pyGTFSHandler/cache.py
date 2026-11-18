"""Cache-folder resolution for pyGTFSHandler (never CWD-relative)."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

CACHE_ENV_VAR = "PYGTFSHANDLER_CACHE_DIR"


def get_cache_dir(
    namespace: str = "",
    cache_dir: Optional[Union[str, Path]] = None,
    create: bool = True,
) -> Path:
    """Resolve a cache folder: explicit ``cache_dir`` (returned as is), else
    ``$PYGTFSHANDLER_CACHE_DIR/<namespace>``, else ``<tempdir>/pygtfshandler-cache/<namespace>``.

    Args:
        namespace: Sub-folder of the cache root.
        cache_dir: Explicit folder; returned as is.
        create: Create the folder if missing.

    Returns:
        The cache folder.
    """
    if cache_dir:
        path = Path(cache_dir)
    else:
        env = os.environ.get(CACHE_ENV_VAR)
        root = Path(env).expanduser() if env else Path(tempfile.gettempdir()) / "pygtfshandler-cache"
        path = root / namespace if namespace else root
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
