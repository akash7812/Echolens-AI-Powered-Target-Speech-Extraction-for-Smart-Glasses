from setuptools import setup, find_packages

setup(
    name="echolens",
    version="1.0.0",
    description="AI-Powered Target Speech Extraction for Smart Glasses (CHiME-10 Task 3)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "echolens = echolens.cli:main",
        ],
    },
)
