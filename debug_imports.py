print("Starting import test...")
import os
import sys
from dotenv import load_dotenv
load_dotenv()

print("Importing FastAPI...")
from fastapi import FastAPI
print("Importing langchain...")
from langchain_openai import ChatOpenAI
print("Importing chromadb...")
import chromadb
print("Importing src.api...")
try:
    from src.api import app
    print("src.api imported successfully!")
except Exception as e:
    print(f"Error importing src.api: {e}")
    import traceback
    traceback.print_exc()
