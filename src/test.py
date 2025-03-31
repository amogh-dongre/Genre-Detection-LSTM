import json

try:
    with open('./genre_mapping.json', 'r') as f:
        mapping = json.load(f)
    print("Genre mapping loaded successfully")
except Exception as e:
    print(f"Error loading genre mapping: {e}")

