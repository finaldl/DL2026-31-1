
from datasets import load_dataset

pets = load_dataset("pcuenq/oxford-pets")
print(pets)

# Kiểm tra số lớp (label) — phải ra 37
labels = pets["train"].unique("label")
print("Số lớp:", len(labels))
from datasets import load_dataset

pets = load_dataset("pcuenq/oxford-pets")
print(pets)

labels = pets["train"].unique("label")
print("Số lớp:", len(labels))

# Chia trainval / test (80/20), giữ tỉ lệ đều giữa các lớp (stratify)
split = pets["train"].train_test_split(
    test_size=0.2, seed=42, stratify_by_column="label"
)
trainval = split["train"]
test = split["test"]

print("Trainval:", len(trainval), "| Test:", len(test))

# Lưu lại ra đĩa để dùng cho các bước sau (sampler, feature extraction)
trainval.save_to_disk("./data/pets_trainval")
test.save_to_disk("./data/pets_test")
print("Đã lưu vào ./data/pets_trainval và ./data/pets_test")