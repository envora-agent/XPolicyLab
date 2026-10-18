"""allenai/Molmo2-4B tool; importing the client does not load model libraries."""

from dataclasses import dataclass
from typing import ClassVar

from envora_sim.tools._vlm import Client as VLMClient
from envora_sim.tools._vlm import Config as VLMConfig


@dataclass(frozen=True)
class Config(VLMConfig):
    model: str = "allenai/Molmo2-4B"
    default_revision: ClassVar[str] = "042abfa7a38879a376cec03d949eff0aefaa0600"
    backend: ClassVar[str] = "molmo2"


class Client(VLMClient):
    config_type = Config


add_arguments = Config.add_arguments
__all__ = ("Client", "Config", "add_arguments")
