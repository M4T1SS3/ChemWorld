"""
Error Budget System for Training

Tracks batch failures and halts training if error rate exceeds threshold.
Provides comprehensive logging for debugging numerical instability.
"""

import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime
import traceback


class ErrorBudget:
    """
    Error budget tracker for training stability.

    Monitors failure rate over a rolling window and triggers emergency
    checkpoint/halt when budget is exceeded.

    Args:
        threshold: Maximum allowed failure rate (default: 0.05 = 5%)
        window_size: Number of batches for rolling window (default: 100)
        log_dir: Directory for failure logs (default: "logs/failures")
    """

    def __init__(
        self,
        threshold: float = 0.05,
        window_size: int = 100,
        log_dir: str = "logs/failures",
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.log_dir = Path(log_dir)

        # Rolling window of success/failure
        self.window = deque(maxlen=window_size)

        # Cumulative statistics
        self.total_batches = 0
        self.total_failures = 0
        self.consecutive_failures = 0

        # Failure records (for comprehensive logging)
        self.failure_records = []

        # Session info
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y-%m-%d_%H-%M-%S")

        # Create log directory
        self.session_log_dir = self.log_dir / self.session_id
        self.session_log_dir.mkdir(parents=True, exist_ok=True)

        print(f"ErrorBudget initialized:")
        print(f"  Threshold: {threshold*100:.1f}%")
        print(f"  Window size: {window_size} batches")
        print(f"  Log directory: {self.session_log_dir}")

    def record_success(self):
        """Record a successful batch."""
        self.window.append(True)
        self.total_batches += 1
        self.consecutive_failures = 0

    def record_failure(
        self,
        batch_idx: int,
        error: Exception,
        batch_data: Optional[Dict] = None,
        intermediate_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Record a failed batch with comprehensive logging.

        Args:
            batch_idx: Index of failed batch
            error: Exception that was raised
            batch_data: Batch data (should include 'smiles' list)
            intermediate_tensors: Any intermediate tensors for debugging
        """
        self.window.append(False)
        self.total_batches += 1
        self.total_failures += 1
        self.consecutive_failures += 1

        # Create failure record
        record = {
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "consecutive_failures": self.consecutive_failures,
        }

        # Add SMILES if available
        if batch_data is not None and "smiles" in batch_data:
            smiles_list = batch_data["smiles"]
            record["smiles"] = smiles_list[:20]  # Limit to first 20
            record["batch_size"] = len(smiles_list)

        # Add tensor diagnostics
        if intermediate_tensors is not None:
            diagnostics = {}
            for name, tensor in intermediate_tensors.items():
                if isinstance(tensor, torch.Tensor):
                    diagnostics[name] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        "has_nan": bool(torch.isnan(tensor).any()),
                        "has_inf": bool(torch.isinf(tensor).any()),
                        "min": float(tensor.min()) if not torch.isnan(tensor).any() else "NaN",
                        "max": float(tensor.max()) if not torch.isnan(tensor).any() else "NaN",
                        "mean": float(tensor.mean()) if not torch.isnan(tensor).any() else "NaN",
                    }
            record["tensor_diagnostics"] = diagnostics

        self.failure_records.append(record)

        # Save individual failure log
        self._save_failure_log(batch_idx, record, intermediate_tensors)

    def _save_failure_log(
        self,
        batch_idx: int,
        record: Dict,
        intermediate_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Save detailed log for a single failure."""
        # Save JSON record
        json_path = self.session_log_dir / f"failure_batch_{batch_idx}.json"
        with open(json_path, 'w') as f:
            json.dump(record, f, indent=2)

        # Save tensors if provided
        if intermediate_tensors is not None:
            tensor_path = self.session_log_dir / f"tensors_batch_{batch_idx}.pt"
            # Filter to only torch tensors
            tensors_to_save = {
                k: v for k, v in intermediate_tensors.items()
                if isinstance(v, torch.Tensor)
            }
            if tensors_to_save:
                torch.save(tensors_to_save, tensor_path)

    def check_budget(self) -> bool:
        """
        Check if error budget has been exceeded.

        Returns:
            True if budget exceeded (should halt), False otherwise
        """
        if len(self.window) < self.window_size:
            # Not enough data yet, don't halt
            return False

        failure_count = sum(1 for x in self.window if not x)
        failure_rate = failure_count / len(self.window)

        return failure_rate > self.threshold

    def get_statistics(self) -> Dict[str, Any]:
        """Get current error budget statistics."""
        if len(self.window) == 0:
            return {
                "total_batches": 0,
                "total_failures": 0,
                "failure_rate": 0.0,
                "window_failure_rate": 0.0,
                "consecutive_failures": 0,
                "budget_exceeded": False,
            }

        window_failures = sum(1 for x in self.window if not x)
        window_rate = window_failures / len(self.window)
        overall_rate = self.total_failures / self.total_batches if self.total_batches > 0 else 0.0

        return {
            "total_batches": self.total_batches,
            "total_failures": self.total_failures,
            "failure_rate": overall_rate,
            "window_failure_rate": window_rate,
            "window_size": len(self.window),
            "consecutive_failures": self.consecutive_failures,
            "budget_exceeded": self.check_budget(),
            "threshold": self.threshold,
        }

    def get_progress_info(self) -> str:
        """Get compact string for progress bar."""
        stats = self.get_statistics()
        rate = stats["window_failure_rate"] * 100

        if stats["budget_exceeded"]:
            return f"failures={stats['total_failures']}/{stats['total_batches']} ({rate:.1f}%) ✗"
        else:
            return f"failures={stats['total_failures']}/{stats['total_batches']} ({rate:.1f}%) ✓"

    def save_final_report(self, checkpoint_path: Optional[Path] = None) -> Path:
        """
        Save comprehensive failure report.

        Args:
            checkpoint_path: Path to emergency checkpoint (if saved)

        Returns:
            Path to the failure report
        """
        stats = self.get_statistics()

        # Create markdown report
        report_path = self.session_log_dir / "FAILURE_REPORT.md"

        with open(report_path, 'w') as f:
            f.write("# Training Failure Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Date:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration:** {(datetime.now() - self.session_start).total_seconds():.1f}s\n\n")

            f.write("## Error Budget Status\n\n")
            f.write(f"- **Threshold:** {self.threshold*100:.1f}%\n")
            f.write(f"- **Window Failure Rate:** {stats['window_failure_rate']*100:.1f}%\n")
            f.write(f"- **Overall Failure Rate:** {stats['failure_rate']*100:.1f}%\n")
            f.write(f"- **Total Batches:** {stats['total_batches']}\n")
            f.write(f"- **Total Failures:** {stats['total_failures']}\n")
            f.write(f"- **Consecutive Failures:** {stats['consecutive_failures']}\n")
            f.write(f"- **Budget Exceeded:** {'✗ YES' if stats['budget_exceeded'] else '✓ NO'}\n\n")

            if checkpoint_path:
                f.write("## Emergency Checkpoint\n\n")
                f.write(f"Checkpoint saved to: `{checkpoint_path}`\n\n")

            f.write("## Recent Failures\n\n")
            recent_failures = self.failure_records[-10:]  # Last 10
            for i, record in enumerate(recent_failures, 1):
                f.write(f"### Failure {i} (Batch {record['batch_idx']})\n\n")
                f.write(f"- **Error:** `{record['error_type']}: {record['error_message']}`\n")
                f.write(f"- **Timestamp:** {record['timestamp']}\n")

                if 'smiles' in record:
                    f.write(f"- **Batch Size:** {record['batch_size']}\n")
                    f.write(f"- **SMILES (first 5):**\n")
                    for smiles in record['smiles'][:5]:
                        f.write(f"  - `{smiles}`\n")

                if 'tensor_diagnostics' in record:
                    f.write("- **Tensor Diagnostics:**\n")
                    for name, diag in record['tensor_diagnostics'].items():
                        nan_flag = "⚠️ NaN" if diag['has_nan'] else ""
                        inf_flag = "⚠️ Inf" if diag['has_inf'] else ""
                        f.write(f"  - `{name}`: shape={diag['shape']} {nan_flag} {inf_flag}\n")

                f.write("\n")

            f.write("## Recommendations\n\n")

            if stats['consecutive_failures'] > 10:
                f.write("⚠️ **High consecutive failures detected!**\n\n")
                f.write("This suggests the model parameters have become corrupted (likely NaN).\n\n")
                f.write("**Recommended actions:**\n")
                f.write("1. Reload from a previous checkpoint before the cascade\n")
                f.write("2. Reduce learning rate by 10x\n")
                f.write("3. Increase gradient clipping (reduce max_norm)\n\n")

            if stats['failure_rate'] > 0.01:
                f.write("**Common failure patterns:**\n")
                error_types = {}
                for record in self.failure_records:
                    error_type = record['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1

                for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                    f.write(f"- `{error_type}`: {count} occurrences\n")
                f.write("\n")

            f.write("## Next Steps\n\n")
            f.write("1. Review individual failure logs in this directory\n")
            f.write("2. Examine tensor dumps for numerical issues\n")
            f.write("3. Check problematic SMILES for common patterns\n")
            f.write("4. Consider reducing batch size or learning rate\n")

        # Also save statistics as JSON
        stats_path = self.session_log_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "statistics": stats,
                "failure_records": self.failure_records,
            }, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Failure report saved to: {report_path}")
        print(f"{'='*80}")

        return report_path
