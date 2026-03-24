"""Root conftest.py — ensures the project root is on sys.path for pytest.

Addresses AUD-006: default ``pytest`` invocation no longer requires
``PYTHONPATH=.`` when this file (or the pyproject.toml pythonpath setting)
is present.
"""
