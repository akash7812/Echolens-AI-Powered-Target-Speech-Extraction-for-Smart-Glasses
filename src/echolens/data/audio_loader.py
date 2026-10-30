import soundfile as sf


def load_audio(file_path):
    """Load audio while preserving microphone channels."""

    audio, sample_rate = sf.read(
        file_path,
        dtype="float32",
        always_2d=True
    )

    # Convert to (channels, samples)
    audio = audio.T

    return audio, sample_rate


def get_audio_info(audio, sample_rate):
    """Return basic audio information."""

    channels, samples = audio.shape

    duration = samples / sample_rate

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "samples": samples
    }