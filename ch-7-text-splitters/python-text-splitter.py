# python_code_splitter_example.py

from langchain_text_splitters import PythonCodeTextSplitter

# 1️⃣ Sample Python code as a string
sample_code = """
import math

# Function to calculate factorial
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Function to calculate square root
def calculate_sqrt(x):
    return math.sqrt(x)

# Main execution
if __name__ == "__main__":
    num = 5
    print("Factorial:", factorial(num))
    print("Square root:", calculate_sqrt(num))
"""

# 2️⃣ Initialize PythonCodeSplitter
splitter = PythonCodeTextSplitter(
    chunk_size=100,      # Max number of characters per chunk
    chunk_overlap=20     # Overlap to preserve context
)

# 3️⃣ Split the code
chunks = splitter.split_text(sample_code)

# 4️⃣ Print each chunk
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
