
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from echolens.models.ps4_extractor import PS4Extractor
    from echolens.models.evaluation import evaluate_sample
except ImportError:
    from src.echolens.models.ps4_extractor import PS4Extractor
    from src.echolens.models.evaluation import evaluate_sample


class EcholensPipeline:
    """
    Main AI pipeline for Echolens Target Speaker Extraction.
    """

    def __init__(self, checkpoint_path, device="cpu"):
        print("[Echolens] Initializing AI pipeline...")

        self.extractor = PS4Extractor(
            checkpoint_path=checkpoint_path,
            device=device
        )

        print("[Echolens] AI pipeline ready.")

    def process_sample(self, sample, output_path=None):
        """
        Extract target speech and evaluate extraction quality.
        """

        required_fields = {
            "mixture",
            "enrollment",
            "target",
            "sample_rate"
        }

        missing_fields = required_fields - set(sample.keys())

        if missing_fields:
            raise ValueError(
                f"Missing sample fields: {missing_fields}"
            )

        mixture = sample["mixture"]
        enrollment = sample["enrollment"]
        target = sample["target"]
        sample_rate = sample["sample_rate"]

        # Step 1: Extract target speaker
        extracted = self.extractor.extract(
            mixture=mixture,
            enrollment=enrollment,
            sample_rate=sample_rate
        )

        # Step 2: Validate extracted audio
        if not np.isfinite(extracted).all():
            raise ValueError(
                "Extracted audio contains invalid values."
            )

        if len(extracted) != len(target):
            raise ValueError(
                "Extracted audio and target lengths do not match."
            )

        # Step 3: Evaluate extraction quality
        metrics = evaluate_sample(
            sample=sample,
            extracted=extracted
        )

        # Step 4: Save extracted audio if requested
        saved_path = None

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            sf.write(
                output_path,
                extracted,
                sample_rate
            )

            saved_path = str(output_path.resolve())

        # Step 5: Return complete results
        return {
            "extracted": extracted,
            "target": target,
            "sample_rate": sample_rate,
            "output_path": saved_path,
            "metrics": metrics
        }
