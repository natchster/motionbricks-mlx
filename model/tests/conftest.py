import pytest

from mlx_helpers import MODELS


@pytest.fixture(params=MODELS)
def model_name(request):
    return request.param
