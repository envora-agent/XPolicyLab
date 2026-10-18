"""Run the allenai/Molmo2-4B single-image worker or one-shot CLI."""

from envora_sim.tools._vlm import main
from envora_sim.tools.molmo2_4b import Config

if __name__ == "__main__":
    raise SystemExit(main(Config))
