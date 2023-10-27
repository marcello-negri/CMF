from abc import ABC

from nflows.transforms.base import Transform
import torch
from typing import Callable


class MonotonicTransform(Transform, ABC):
    def __init__(self, num_iterations=25, lim=10):
        self.num_iterations = num_iterations
        self.lim = lim
        super(MonotonicTransform, self).__init__()

    def bisection_inverse(self, z, context=None):
        x_max = torch.ones_like(z) * self.lim
        x_min = -torch.ones_like(z) * self.lim

        z_max, _ = self.forward(x_max, context)
        z_min, _ = self.forward(x_min, context)

        diff = z - z_max

        idx_maxdiff = torch.argmax(diff)
        maxdiff = diff.flatten()[idx_maxdiff]

        diff = z - z_min
        idx_mindiff = torch.argmin(diff)
        mindiff = diff.flatten()[idx_mindiff]

        if maxdiff > 0:
            ratio = (maxdiff + z_max.flatten()[idx_maxdiff]) / z_max.flatten()[idx_maxdiff]
            x_max = x_max * 1.2 * ratio
            z_max, _ = self.forward(x_max, context)

        if mindiff < 0:
            ratio = (mindiff + z_min.flatten()[idx_mindiff]) / z_min.flatten()[idx_mindiff]
            x_min = x_min * 1.2 * ratio
            z_min, _ = self.forward(x_min, context)

        # Old inversion by binary search
        for i in range(self.num_iterations):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, context)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min

        x = (x_max + x_min) / 2
        return x, -self.forward_logabsdet(x, context=context).squeeze()

    def forward_logabsdet(self, inputs, context=None):
        _, logabsdet = self.forward(inputs=inputs, context=context)
        return logabsdet

    def inverse(self, inputs, context=None):
        return self.bisection_inverse(inputs, context=context)
