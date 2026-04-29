import os
from dotenv import load_dotenv
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Read PDF
doc = fitz.open("basic-text.pdf")
text = ""
for page in doc:
    text += page.get_text()

# Chunk it
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")

# Test Groq connection
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "tell me who you are"}]
)
print(response.choices[0].message.content)