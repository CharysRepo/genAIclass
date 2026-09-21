from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train", streaming="True")
print(dataset)

for item in dataset:
    print(item)
