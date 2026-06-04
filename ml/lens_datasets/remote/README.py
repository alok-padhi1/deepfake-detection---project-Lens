"""
LENS ML — Remote Dataset Strategy
────────────────────────────────────────────────────────────────────
Option A: HuggingFace Hub Streaming   (GenImage — already on HF)
Option B: WebDataset + S3/GCS         (DFDC, FaceForensics++)
Option C: HuggingFace Hub + S3 hybrid (ASVspoof, Synthetic)

No local download required. Data streams during training.
────────────────────────────────────────────────────────────────────

Requirements by option:
  Option A: pip install datasets huggingface_hub
  Option B: pip install webdataset boto3           (S3)
            pip install webdataset google-cloud-storage  (GCS)
  Option C: both

Storage needed locally:
  - ZERO for raw dataset
  - ~5GB for preprocessed frame cache (optional, speeds up training)
"""
