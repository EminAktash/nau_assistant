import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the scraped content
with open('na_edu_content.json', 'r') as f:
    content = json.load(f)

# Create chunks of text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

documents = []
for url, text in content.items():
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        documents.append({
            "content": chunk,
            "source": url
        })

# Save the chunks to a file
with open('na_edu_chunks.json', 'w') as f:
    json.dump(documents, f)

print(f"Created {len(documents)} chunks from {len(content)} pages")