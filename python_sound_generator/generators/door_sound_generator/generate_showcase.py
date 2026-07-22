"""Render a compact bank and listening preview for the door generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .engine import AudioBuffer, generate_door_sound, write_wav


SHOWCASE = (
    ("heavy_wood_open", "open", "heavy_wood", 0.82, 0.72, 1201),
    ("heavy_wood_close", "close", "heavy_wood", 0.88, 0.62, 1217),
    ("heavy_wood_slam", "slam", "heavy_wood", 1.00, 0.52, 1237),
    ("old_hollow_wood_creak", "creak", "hollow_wood", 0.78, 0.96, 1259),
    ("light_wood_latch", "latch", "light_wood", 0.72, 0.35, 1283),
    ("steel_door_close", "close", "metal", 0.90, 0.48, 1301),
)


def generate_showcase(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[np.ndarray] = []
    artifacts: list[dict[str, object]] = []
    paths: list[Path] = []
    sample_rate = 44_100

    for name, action, material, intensity, age, seed in SHOWCASE:
        audio = generate_door_sound(
            action=action,
            material=material,
            intensity=intensity,
            age=age,
            room=0.10,
            seed=seed,
            sample_rate=sample_rate,
            stereo=False,
        )
        path = write_wav(output_dir / f"{name}.wav", audio)
        signal = np.asarray(audio.channels[0], dtype=float)
        rendered.append(signal)
        artifacts.append(
            {
                "file": path.name,
                "action": action,
                "material": material,
                "seed": seed,
                "sources": list(audio.sources),
                "duration_seconds": round(audio.duration, 4),
                "peak_dbfs": round(
                    20.0 * np.log10(max(1e-12, float(np.max(np.abs(signal))))), 3
                ),
                "rms_dbfs": round(
                    20.0 * np.log10(max(1e-12, float(np.sqrt(np.mean(signal * signal))))),
                    3,
                ),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        paths.append(path)

    gap = np.zeros(round(0.32 * sample_rate), dtype=float)
    preview_parts: list[np.ndarray] = []
    for index, signal in enumerate(rendered):
        preview_parts.append(signal)
        if index + 1 < len(rendered):
            preview_parts.append(gap)
    preview = np.concatenate(preview_parts)
    preview_path = write_wav(
        output_dir / "door_generator_preview.wav",
        AudioBuffer(sample_rate, (preview.tolist(),)),
    )
    paths.append(preview_path)

    manifest = {
        "generator": "sample-driven NumPy/SciPy door Foley variation engine",
        "format": "mono 44100 Hz signed 16-bit PCM WAV",
        "preview_order": [item[0] for item in SHOWCASE],
        "artifacts": artifacts,
        "preview": preview_path.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "doors" / "showcase",
    )
    arguments = parser.parse_args()
    paths = generate_showcase(arguments.output_dir.resolve())
    print(f"Generated {len(SHOWCASE)} door sounds and one preview:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
