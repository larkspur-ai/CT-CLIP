from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FakeGenerateLatentsOutput:
    latents: np.ndarray

    @property
    def texts(self):
        return [self.latents]


@dataclass(frozen=True)
class FakeAccelerator:
    device: str = "fake_device"


class FakeCtClip:
    accelerator = FakeAccelerator()

    @staticmethod
    def generate_latents(text: str) -> FakeGenerateLatentsOutput:
        """Generate a random latent vector for the given text.

        Random values are generated with a seed based on the hash of the text,
        so the same text will always produce the same latent vector.
        """
        text_hash = abs(hash(text))
        return FakeGenerateLatentsOutput(
            np.random.default_rng(seed=text_hash).random(size=(1, 512))
        )
