import pytest
import numpy as np
import warnings

import torch
import jax
import jax.numpy as jnp

from models_flax import AlexNetFlax, GoogleNetFlax
from models_pytorch import AlexNetPyTorch, GoogleNetPyTorch


def build_pytorch_model(ModelClass):
    model = ModelClass()
    model.to('cuda')
    model.eval()

    x = np.random.rand(16, 3, 224, 224).astype(np.float32)
    x = torch.from_numpy(x).to('cuda')

    return model, x

def build_flax_model(ModelClass):
    model = AlexNetFlax()

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key1, (16, 224, 224, 3))
    weight = model.init(key2, x) # Initialization cal

    return model, weight, x

@pytest.mark.benchmark(
    group="AlexNet",
    warmup=True
)
def test_AlexNetPytorch(benchmark):
    model, x = build_pytorch_model(AlexNetPyTorch)

    def run(_x):
        with torch.no_grad():
            return model(_x)

    benchmark.pedantic(run, args=(x,), warmup_rounds=2, iterations=20, rounds=3)

@pytest.mark.benchmark(
    group="AlexNet",
    warmup=True
)
def test_AlexNetFlax(benchmark):
    model, weight, x = build_flax_model(AlexNetFlax)

    @jax.jit
    def run(_x):
        y = model.apply(weight, _x)
        jax.block_until_ready(y)
        return y

    benchmark.pedantic(run, args=(x,), warmup_rounds=2, iterations=20, rounds=3)

@pytest.mark.benchmark(
    group="GoogleNet",
    warmup=True
)
def test_GoogleNetPytorch(benchmark):
    model, x = build_pytorch_model(GoogleNetPyTorch)

    def run(_x):
        with torch.no_grad():
            return model(_x)

    benchmark.pedantic(run, args=(x,), warmup_rounds=2, iterations=20, rounds=3)

@pytest.mark.benchmark(
    group="GoogleNet",
    warmup=True
)
def test_GoogleNetFlax(benchmark):
    model, weight, x = build_flax_model(GoogleNetFlax)

    @jax.jit
    def run(_x):
        y = model.apply(weight, _x)
        jax.block_until_ready(y)
        return y

    benchmark.pedantic(run, args=(x,), warmup_rounds=2, iterations=20, rounds=3)


if __name__ == "__main__":
    pytest.main(['-v', __file__])
