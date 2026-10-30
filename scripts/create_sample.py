from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5

# Generate time values
time = np.arange(SAMPLE_RATE * DURATION) / SAMPLE_RATE

# Create two audio channels
channel_1 = 0.3 * np.sin(2 * np.pi * 440 * time)
channel_2 = 0.3 * np.sin(2 * np.pi * 660 * time)

# Combine both channels
audio = np.column_stack((channel_1, channel_2))

# Set output path
project_root = Path(__file__).resolve().parents[1]
output_path = project_root / "data" / "raw" / "sample.wav"

# Create folder if necessary
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save audio
sf.write(output_path, audio, SAMPLE_RATE)

print("Sample audio created successfully!")
print(f"Saved to: {output_path}")