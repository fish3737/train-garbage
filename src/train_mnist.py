import argparse
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import numpy as np

from src.datasets import create_mnist_dataset, download_mnist
from src.models import LeNet5
from src.utils import Timer, ensure_dir, save_json, set_seed


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


def train(args):
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    set_seed(args.seed)
    download_mnist(args.data_dir)

    train_ds = create_mnist_dataset(args.data_dir, train=True, batch_size=args.batch_size, max_samples=args.max_train_samples)
    val_ds = create_mnist_dataset(args.data_dir, train=False, batch_size=args.batch_size, max_samples=args.max_val_samples)

    network = LeNet5(num_classes=10)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(network.trainable_params(), learning_rate=args.lr)
    train_cell = nn.TrainOneStepCell(nn.WithLossCell(network, loss_fn), optimizer)
    train_cell.set_train()

    history = []
    with Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            losses = []
            for images, labels in train_ds.create_tuple_iterator():
                loss = train_cell(images, labels)
                losses.append(float(loss.asnumpy()))
            metrics = evaluate(network, val_ds)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": metrics["loss"],
                "val_accuracy": metrics["accuracy"],
            }
            history.append(row)
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
            )

    ensure_dir(str(Path(args.ckpt).parent))
    ms.save_checkpoint(network, args.ckpt)
    result = {
        "task": "mnist_pretrain",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_seconds": timer.seconds,
        "checkpoint": args.ckpt,
        "history": history,
        "final": history[-1],
    }
    save_json(result, args.out)
    print(f"saved checkpoint: {args.ckpt}")
    print(f"saved metrics: {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST and save a pretrained checkpoint.")
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--ckpt", default="checkpoints/lenet5_mnist.ckpt")
    parser.add_argument("--out", default="outputs/mnist_metrics.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Use a small subset for quick tests; 0 means full MNIST.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Use a small subset for quick tests; 0 means full MNIST.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-target", default="CPU")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
