# # Create a small script named check_db.py
# import chromadb
# client = chromadb.PersistentClient(path="./data/chroma_db")
# print("Existing collections:", client.list_collections())


import chromadb
from pathlib import Path

# 1. Get the directory where THIS script is located
current_file_dir = Path(__file__).parent.resolve()

# 2. Go UP one level to the project root, then into 'data/chroma_db'
db_path = current_file_dir.parent / "data" / "chroma_db"

print(f"🔍 Searching for DB at: {db_path}")

if not db_path.exists():
    print(f"❌ Folder not found at {db_path}. Check your path logic.")
else:
    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()
    
    if not collections:
        print("❓ No collections found.")
    else:
        print(f"✅ Found {len(collections)} collection(s):")
        for col in collections:
            # Note: In newer Chroma, list_collections returns objects with .name
            print(f"  - Name: '{col.name}' | Chunks: {col.count()}")