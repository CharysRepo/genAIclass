from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
print(dataset)

# Explore the metadata of a dataset
print(f"Available splits: {list(dataset.keys())}")
print(f"Features (Columns): {dataset['train'].features}")
print(f"Total training examples: {len(dataset['train'])}")

train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Access a specific slice or sample from the dataset
first_review = train_data[0]
print(f"First training item:\n{first_review}")

# Filter data dynamically
positive_reviews = train_data.filter(lambda x: x["label"] == 1)
print(f"Number of positive reviews in train split: {len(positive_reviews)}")

# Transform and preprocess data using .map()
def uppercase_text(example):
    example["text"] = example["text"].upper()
    return example

transformed_dataset = dataset.map(uppercase_text)
print("Original text sample:", dataset["train"][0]["text"][:60])
print("Transformed text sample:", transformed_dataset["train"][0]["text"][:60])

# Convert to standard formats like Pandas DataFrame
df = transformed_dataset["train"].to_pandas()
print(f"Successfully converted to Pandas DataFrame. Shape: {df.shape}")
print(df.head(2))
