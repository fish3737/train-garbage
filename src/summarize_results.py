import json
from pathlib import Path


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    mnist = load("outputs/mnist_metrics.json")
    garbage = load("outputs/garbage_metrics.json")
    improved = load("outputs/improved_garbage_metrics.json")

    lines = ["# 实验结果摘要", ""]
    if mnist:
        final = mnist["final"]
        lines.append("## MNIST 预训练")
        lines.append(f"- 训练轮数：{mnist['epochs']}")
        lines.append(f"- 验证准确率：{final['val_accuracy']:.4f}")
        lines.append(f"- 训练耗时：{mnist['train_seconds']:.2f} 秒")
        lines.append(f"- 保存权重：`{mnist['checkpoint']}`")
        lines.append("")

    if garbage:
        lines.append("## 垃圾分类微调")
        lines.append(f"- 类别数：{len(garbage['class_to_idx'])}")
        for exp in garbage["experiments"]:
            final = exp["final"]
            memory = exp.get("memory_peak_mb")
            memory_text = f"，峰值内存 {memory:.2f} MB" if memory else ""
            lines.append(
                f"- {exp['mode']}：验证准确率 {final['val_accuracy']:.4f}，"
                f"训练耗时 {exp['train_seconds']:.2f} 秒{memory_text}，权重 `{exp['checkpoint']}`"
            )
        lines.append("")

    if improved:
        final = improved["final"]
        lines.append("## 改进模型")
        lines.append(f"- 模型：{improved['model']}")
        lines.append(f"- 输入：RGB {improved['image_size']}x{improved['image_size']}")
        lines.append(
            f"- 验证准确率：{final['val_accuracy']:.4f}，训练耗时 {improved['train_seconds']:.2f} 秒，"
            f"峰值内存 {improved.get('memory_peak_mb', 0):.2f} MB"
        )
        lines.append(f"- 保存权重：`{improved['checkpoint']}`")
        lines.append("")

    out = Path("outputs/result_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
