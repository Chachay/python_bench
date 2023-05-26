from typing import Any

import jax
import jax.numpy as jnp

from flax import linen as nn

__all__ = [
    "AlexNetFlax",
    "GoogleNetFlax"
]

#-----------------
# AlexNet
#-----------------
class AlexNetFlax(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(96, (11, 11), strides=(4, 4), name='conv1')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = nn.Conv(256, (5, 5), padding=((2, 2),(2, 2)), name='conv2')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2 ,2))

        x = nn.Conv(384, (3, 3), padding=((1, 1),(1, 1)), name='conv3')(x)
        x = nn.relu(x)

        x = nn.Conv(384, (3, 3), padding=((1, 1),(1 ,1)), name='conv4')(x)
        x = nn.relu(x)

        x = nn.Conv(256, (3, 3), padding=((1, 1),(1, 1)), name='conv5')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        #y = y.reshape((y.shape[0], -1))
        #y = nn.Dense(4096)(y)
        #y = nn.Dense(100)(y)

        return x

#-----------------
# GoogLeNet
#-----------------
class InceptionBlock(nn.Module):
    c_red : dict  # Dictionary of reduced dimensionalities with keys "1x1", "3x3", "5x5", and "max"
    c_out : dict  # Dictionary of output feature sizes with keys "1x1", "3x3", "5x5", and "max"

    @nn.compact
    def __call__(self, x):
        # 1x1 convolution branch
        x_1x1 = nn.Conv(self.c_out["1x1"], kernel_size=(1, 1), use_bias=False)(x)
        x_1x1 = nn.relu(x_1x1)

        # 3x3 convolution branch
        x_3x3 = nn.Conv(self.c_red["3x3"], kernel_size=(1, 1), use_bias=False)(x)
        x_3x3 = nn.relu(x_3x3)
        x_3x3 = nn.Conv(self.c_out["3x3"], kernel_size=(3, 3), use_bias=False)(x_3x3)
        x_3x3 = nn.relu(x_3x3)

        # 5x5 convolution branch
        x_5x5 = nn.Conv(self.c_red["5x5"], kernel_size=(1, 1), use_bias=False)(x)
        x_5x5 = nn.relu(x_5x5)
        x_5x5 = nn.Conv(self.c_out["5x5"], kernel_size=(5, 5), use_bias=False)(x_5x5)
        x_5x5 = nn.relu(x_5x5)

        # Max-pool branch
        x_max = nn.max_pool(x, (3, 3), strides=(2, 2))
        x_max = nn.Conv(self.c_out["max"], kernel_size=(1, 1), use_bias=False)(x)
        x_max = nn.relu(x_max)

        x_out = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_max], axis=-1)
        return x_out

class GoogleNetFlax(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(64, kernel_size=(3, 3), use_bias=False)(x)
        x = nn.relu(x)

        x = InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8})(x)
        x = InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12})(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))  # 32x32 => 16x16
        x = InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12})(x)
        x = InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16})(x)
        x = InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16})(x)
        x = InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24})(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2))  # 16x16 => 8x8
        x = InceptionBlock(c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16})(x)
        x = InceptionBlock(c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16})(x)

        # x = x.mean(axis=(1, 2))
        # x = nn.Dense(100)(x)
        return x
