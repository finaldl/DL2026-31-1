import random
from datasets import load_from_disk

def make_kshot_splits(dataset, k_values, seed, label_col="label"):
    """
    Chia dataset thành các tập con theo từng mức k (số ảnh/lớp),
    đảm bảo: cùng seed -> luôn ra kết quả giống nhau, và k nhỏ nằm
    gọn trong k lớn hơn (nested).
    """
    rng = random.Random(seed)

    indices_by_label = {}
    for idx, label in enumerate(dataset[label_col]):
        indices_by_label.setdefault(label, []).append(idx)

    for label in indices_by_label:
        rng.shuffle(indices_by_label[label])

    k_values_sorted = sorted(
        [k for k in k_values if k != "all"],
        key=lambda x: int(x)
    )

    splits = {}
    for k in k_values_sorted:
        k = int(k)
        selected = []
        for label, idxs in indices_by_label.items():
            take = idxs[:k]
            if len(take) < k:
                print(f"[Cảnh báo] Lớp {label} chỉ có {len(take)} ảnh, ít hơn k={k}")
            selected.extend(take)
        splits[k] = selected

    if "all" in k_values:
        splits["all"] = list(range(len(dataset)))

    return splits


if __name__ == "__main__":
    trainval = load_from_disk("./data/pets_trainval")

    k_values = [1, 2, 5, 10, 25, 50, "all"]
    seed = 0

    splits = make_kshot_splits(trainval, k_values, seed)

    for k, idxs in splits.items():
        print(f"k={k}: {len(idxs)} ảnh")

    set5 = set(splits[5])
    set10 = set(splits[10])
    print("k=5 nằm trong k=10:", set5.issubset(set10))