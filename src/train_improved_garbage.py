import argparse
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import numpy as np

from src.datasets import (
    create_rgb_garbage_dataset,
    download_garbage_zip,
    extract_archive,
    prepare_image_folder,
)
from src.models import ImprovedGarbageCNN
from src.utils import MemoryTracker, Timer, ensure_dir, save_json, set_seed


def evaluate(network, dataset):
    network.set_train(False)
    total = 0
    correct = 0
    losses = []
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    for images, labels in dataset.create_tuple_iterator():
        logits = network(images)
        loss = loss_fn(logits, labels)
        preds = np.argmax(logits.asnumpy(), axis=1)
        y = labels.asnumpy()
        correct += int((preds == y).sum())
        total += len(y)
        losses.append(float(loss.asnumpy()))
    return {"loss": float(np.mean(losses)), "accuracy": correct / total}


def main(args):
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    set_seed(args.seed)

    if args.download:
        download_garbage_zip(args.zip_path)
    if Path(args.zip_path).exists():
        extract_archive(args.zip_path, args.raw_dir)

    class_to_idx = prepare_image_folder(args.raw_dir, args.prepared_dir, args.train_ratio, args.seed)
    train_ds = create_rgb_garbage_dataset(
        str(Path(args.prepared_dir) / "train"),
        class_to_idx,
        batch_size=args.batch_size,
        train=True,
        image_size=args.image_size,
    )
    val_ds = create_rgb_garbage_dataset(
        str(Path(args.prepared_dir) / "val"),
        class_to_idx,
        batch_size=args.batch_size,
        train=False,
        image_size=args.image_size,
    )

    network = ImprovedGarbageCNN(num_classes=len(class_to_idx))
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(network.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    train_cell = nn.TrainOneStepCell(nn.WithLossCell(network, loss_fn), optimizer)
    train_cell.set_train()

    history = []
    with Timer() as timer, MemoryTracker() as memory:
        for epoch in range(1, args.epochs + 1):
            network.set_train(True)
            losses = []
            for images, labels in train_ds.create_tuple_iterator():
                loss = train_cell(images, labels)
                losses.append(float(loss.asnumpy()))
                memory.update()
            metrics = evaluate(network, val_ds)
            memory.update()
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
                "memory_peak_mb": memory.peak_mb,
            }
            history.append(row)
            print(
                f"improved epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
            )

    ensure_dir(str(Path(args.ckpt).parent))
    ms.save_checkpoint(network, args.ckpt)
    result = {
        "task": "garbage_improved",
        "model": "ImprovedGarbageCNN",
        "class_to_idx": class_to_idx,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "train_seconds": timer.seconds,
        "memory_peak_mb": memory.peak_mb,
        "checkpoint": args.ckpt,
        "history": history,
        "final": history[-1],
    }
    save_json(result, args.out)
    print(f"saved checkpoint: {args.ckpt}")
    print(f"saved metrics: {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an improved RGB CNN on garbage classification.")
    parser.add_argument("--zip-path", default="data/data_en.zip")
    parser.add_argument("--raw-dir", default="data/garbage_raw")
    parser.add_argument("--prepared-dir", default="data/garbage_split")
    parser.add_argument("--ckpt", default="checkpoints/improved_garbage_cnn.ckpt")
    parser.add_argument("--out", default="outputs/improved_garbage_metrics.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-target", default="CPU")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
