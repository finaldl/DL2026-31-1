from datasets import load_dataset

eurosat = load_dataset("tanganke/eurosat")
print(eurosat)

train = eurosat["train"]
test = eurosat["test"]

labels = train.unique("label")
print("Số lớp EuroSAT:", len(labels))
print("Train:", len(train), "| Test:", len(test))

train.save_to_disk("./data/eurosat_train")
test.save_to_disk("./data/eurosat_test")
print("Đã lưu vào ./data/eurosat_train và ./data/eurosat_test")