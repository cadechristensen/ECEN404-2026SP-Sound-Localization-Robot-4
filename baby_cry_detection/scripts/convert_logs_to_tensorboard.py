"""Convert training_history.json from past runs into TensorBoard event files."""

import json
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def convert_run(run_dir: Path) -> bool:
    history_file = run_dir / "training_history.json"
    if not history_file.exists():
        print(f"  Skipped {run_dir.name} — no training_history.json")
        return False

    tb_dir = run_dir / "tensorboard"
    if tb_dir.exists() and any(tb_dir.iterdir()):
        print(f"  Skipped {run_dir.name} — TensorBoard logs already exist")
        return False

    with open(history_file) as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    learning_rates = history.get("learning_rates", [])

    if not train_loss:
        print(f"  Skipped {run_dir.name} — empty history")
        return False

    writer = SummaryWriter(log_dir=str(tb_dir))

    for epoch in range(len(train_loss)):
        if epoch < len(train_loss):
            writer.add_scalar("Loss/Train", train_loss[epoch], epoch + 1)
        if epoch < len(val_loss):
            writer.add_scalar("Loss/Validation", val_loss[epoch], epoch + 1)
        if epoch < len(train_acc):
            writer.add_scalar("Accuracy/Train", train_acc[epoch], epoch + 1)
        if epoch < len(val_acc):
            writer.add_scalar("Accuracy/Validation", val_acc[epoch], epoch + 1)
        if epoch < len(learning_rates):
            writer.add_scalar("Learning Rate", learning_rates[epoch], epoch + 1)

    writer.close()
    print(f"  Converted {run_dir.name} — {len(train_loss)} epochs")
    return True


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    run_dirs = sorted(results_dir.glob("train_*"))
    print(f"Found {len(run_dirs)} training runs\n")

    converted = 0
    for run_dir in run_dirs:
        if run_dir.is_dir() and convert_run(run_dir):
            converted += 1

    print(f"\nDone — converted {converted}/{len(run_dirs)} runs")
    print(f"\nTo view in VS Code:")
    print(f"  Ctrl+Shift+P -> 'Python: Launch TensorBoard'")
    print(f"  Log directory: {results_dir}")


if __name__ == "__main__":
    main()
