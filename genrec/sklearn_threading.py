import os
from contextlib import contextmanager


THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
DEFAULT_SKLEARN_THREAD_LIMIT = 64


def _coerce_thread_limit(thread_limit):
    if thread_limit is None:
        return DEFAULT_SKLEARN_THREAD_LIMIT
    thread_limit = int(thread_limit)
    if thread_limit <= 0:
        return None
    return thread_limit


def _is_safe_thread_env_value(value, thread_limit):
    if value is None:
        return False
    try:
        value = int(value)
    except (TypeError, ValueError):
        return False
    return 0 < value <= thread_limit


def configure_sklearn_threads(thread_limit=None):
    thread_limit = _coerce_thread_limit(thread_limit)
    if thread_limit is None:
        return None

    for env_var in THREAD_ENV_VARS:
        if not _is_safe_thread_env_value(os.environ.get(env_var), thread_limit):
            os.environ[env_var] = str(thread_limit)
    return thread_limit


@contextmanager
def limit_sklearn_threads(thread_limit):
    thread_limit = _coerce_thread_limit(thread_limit)
    if thread_limit is None:
        yield
        return

    configure_sklearn_threads(thread_limit)

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return

    with threadpool_limits(limits=thread_limit):
        yield
