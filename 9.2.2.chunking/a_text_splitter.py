from langchain_text_splitters import CharacterTextSplitter

text = "The generative ai technology showcases the potential of AI for businesses, individuals and society."
print(text)

text_splitter = CharacterTextSplitter(
    chunk_size=30, chunk_overlap=0, length_function=len, separator=""
)
chunks = text_splitter.split_text(text)
print("Total Chunks without overlap:", len(chunks))
for chunk in chunks:
    print(len(chunk), chunk)
print()

text_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=5, separator="")
chunks = text_splitter.split_text(text)
print("Total Chunks with overlap:", len(chunks))
for chunk in chunks:
    print(len(chunk), chunk)
