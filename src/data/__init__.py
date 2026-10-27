# Mô tả: Khởi tạo gói src.data chứa các hợp đồng dữ liệu và pipeline tiền xử lý.
"""Shared dataset contracts."""

from .schema import CLASS_NAMES, ManifestRecord, PipelineError

__all__ = ["CLASS_NAMES", "ManifestRecord", "PipelineError"]
