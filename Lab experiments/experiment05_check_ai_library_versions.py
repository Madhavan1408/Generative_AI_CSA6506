"""Experiment 5: Verify installed versions of selected AI libraries."""
from importlib.metadata import version, PackageNotFoundError

libraries = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "torch",
    "transformers",
]

for library in libraries:
    try:
        print(f"{library}: {version(library)}")
    except PackageNotFoundError:
        print(f"{library}: Not installed")
