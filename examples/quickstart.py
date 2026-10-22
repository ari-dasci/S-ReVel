"""Runnable end-to-end example of ReVel.

No dataset and no download: the black-box model is a small CNN with fixed random
weights and the instance is a random image, so the script runs anywhere torch is
installed. It fits a LIME explanation and scores it with the five ReVel metrics.

Because the model is randomly initialised, the numbers it prints mean nothing on
their own: what it demonstrates is that the pipeline runs end to end and what the
input/output contract of each piece is. Swap in your own model, instance and
segmentation to get scores worth reading.

Run from the repository root:

    python examples/quickstart.py
"""

import numpy as np
import torch
import torch.nn as nn

from ReVel import ReVel
from ReVel.LLEs import get_xai_model
from ReVel.perturbations import get_perturbation

IMG_SIZE = 32
N_CLASSES = 4
SEGMENT_GRID = 4

torch.manual_seed(0)
np.random.seed(0)


class TinyNet(nn.Module):
    """Stand-in for the black-box classifier: 3x32x32 in, N_CLASSES logits out."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


model = TinyNet().eval()


def classify(image) -> torch.Tensor:
    """Image (HxWxC or 1xCxHxW, numpy or tensor) -> logits of shape (1, C)."""
    if isinstance(image, np.ndarray):
        image = torch.as_tensor(image, dtype=torch.float32)
    image = image.float()
    if image.dim() == 3:
        image = image.permute(2, 0, 1)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    with torch.no_grad():
        return model(image)


# 1. How the image is split into segments and how segments are occluded.
perturbation = get_perturbation(
    name="square",
    dim=SEGMENT_GRID,
    num_classes=N_CLASSES,
    final_size=(IMG_SIZE, IMG_SIZE),
)

# 2. The explanation method under evaluation.
explainer = get_xai_model(
    name="LIME",
    perturbation=perturbation,
    max_examples=100,
    sigma=3,
)

# 3. The instance to explain, and its segmentation.
instance = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
segments = perturbation.segmentation_fn(instance)

# 4. The two black-box views ReVel needs.
def model_f(image) -> torch.Tensor:
    """Input space: an image already perturbed -> logits."""
    return classify(image)


def model_g(X) -> torch.Tensor:
    """Feature space: a binary vector over segments -> logits of the occluded image."""
    x = torch.as_tensor(instance, dtype=torch.float32)
    neutral = perturbation.fn_neutral_image(x)
    avoided = [i for i in range(len(X)) if X[i] == 0]
    occluded = perturbation.perturbation(x, neutral, segments=segments, indexes=avoided)
    return classify(occluded)


revel = ReVel(
    model_f=model_f,
    model_g=model_g,
    instance=instance,
    lle=explainer,
    n_classes=N_CLASSES,
    segments=segments,
)

scores = revel.evaluate(times=2)

print("Per-run scores:")
print(scores.round(4).to_string(index=False))
print("\nMean over runs:")
print(scores.mean().round(4).to_string())
