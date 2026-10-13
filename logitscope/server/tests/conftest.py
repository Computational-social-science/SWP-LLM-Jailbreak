import pytest

from logitscope.tinymodel import build_tiny_bundle


@pytest.fixture(scope="session")
def tiny():
    return build_tiny_bundle(seed=0)
