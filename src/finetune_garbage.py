import argparse
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import numpy as np

from src.datasets import (
    create_garbage_dataset,
    download_garbage_zip,
    extract_archive,
    prepare_image_folder,
)
from src.models import LeNet5, freeze_feature_extractor
from src.utils import MemoryTracker, Timer, ensure_dir, save_json, set_seed


def load_pretrained_except_classifier(network, ckpt_path: str):
    params = ms.load_checkpoint(ckpt_path)
    filtered = {k: v for k, v in params.items() if not k.startswith("fc3.")}
    not_loaded = ms.load_param_into_net(network, filtered, strict_load=False)
    return not_loaded


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


def run_experiment(args, mode: str, class_to_idx):
    train_ds = create_garbage_dataset(
        str(Path(args.prepared_dir) / "train"),
        class_to_idx,
        batch_size=args.batch_size,
        train=True,
        image_size=args.image_size,
    )
    val_ds = create_garbage_dataset(
        str(Path(args.prepared_dir) / "val"),
        class_to_idx,
        batch_size=args.batch_size,
        train=False,
        image_size=args.image_size,
    )

    network = LeNet5(num_classes=len(class_to_idx))
    load_pretrained_except_classifier(network, args.pretrained_ckpt)
    if mode == "frozen":
        freeze_feature_extractor(network)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(network.trainable_params(), learning_rate=args.lr)
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
                f"{mode} epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
            )

    ckpt = args.ckpt_frozen if mode == "frozen" else args.ckpt_full
    ensure_dir(str(Path(ckpt).parent))
    ms.save_checkpoint(network, ckpt)
    return {
        "mode": mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_seconds": timer.seconds,
        "memory_peak_mb": memory.peak_mb,
        "checkpoint": ckpt,
        "history": history,
        "final": history[-1],
    }


def main(args):
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    set_seed(args.seed)

    if args.download:
        download_garbage_zip(args.zip_path)
    if Path(args.zip_path).exists():
        extract_archive(args.zip_path, args.raw_dir)

    class_to_idx = prepare_image_folder(args.raw_dir, args.prepared_dir, args.train_ratio, args.seed)
    print(f"class count: {len(class_to_idx)}")
    print("classes:", ", ".join(class_to_idx.keys()))

    results = {
        "task": "garbage_transfer",
        "pretrained_checkpoint": args.pretrained_ckpt,
        "class_to_idx": class_to_idx,
        "experiments": [],
    }
    modes = ["frozen", "full"] if args.mode == "both" else [args.mode]
    for mode in modes:
        results["experiments"].append(run_experiment(args, mode, class_to_idx))

    save_json(results, args.out)
    print(f"saved metrics: {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MNIST pretrained LeNet-5 on garbage classification.")
    parser.add_argument("--zip-path", default="data/data_en.zip")
    parser.add_argument("--raw-dir", default="data/garbage_raw")
    parser.add_argument("--prepared-dir", default="data/garbage_split")
    parser.add_argument("--pretrained-ckpt", default="checkpoints/lenet5_mnist.ckpt")
    parser.add_argument("--ckpt-frozen", default="checkpoints/lenet5_garbage_frozen.ckpt")
    parser.add_argument("--ckpt-full", default="checkpoints/lenet5_garbage_full.ckpt")
    parser.add_argument("--out", default="outputs/garbage_metrics.json")
    parser.add_argument("--mode", choices=["frozen", "full", "both"], default="both")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-target", default="CPU")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
