import logging
from functools import partial, wraps
from flask import request


class HealthCheckFilter(logging.Filter):
    """Filter for logging output"""

    def __init__(self, path, name=''):
        """Class constructor.
        We pass 'path' argument to instance  which is
        used by to filter logging for Flask routes.
        """
        self.path = path
        super().__init__(name)

    def filter(self, record):
        """Main filter function.
        We add a space after path here to ensure subpaths
        are not unintentionally excluded from logging"""
        return f"{self.path} " not in record.getMessage()


def disable_logging(func=None, *args, **kwargs):
    """Disable log messages for werkzeug log handler
    for a specific Flask routes.

    :param (function) func: wrapped function
    :param (list) args: decorator arguments
    :param (dict) kwargs: decorator keyword arguments
    :return (function) wrapped function
    """
    _logger = 'werkzeug'
    if not func:
        return partial(disable_logging, *args, **kwargs)

    @wraps(func)
    def wrapper(*args, **kwargs):
        path = request.environ['PATH_INFO']
        log = logging.getLogger(_logger)
        log.addFilter(HealthCheckFilter(path))
        return func(*args, **kwargs)
    return wrapper