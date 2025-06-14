import os
import json

INPUT_DIR = "/media/lucid/hitachi/MB204IR/WPI_60K/extracted"

def check_files():
    """Safely examine file contents without slicing"""
    sample_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")][:3]  # Check first 3 files
    
    for f in sample_files:
        file_path = os.path.join(INPUT_DIR, f)
        print(f"\n=== Examining: {f} ===")
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                print(f"Data type: {type(data)}")
                
                if isinstance(data, list):
                    print("First 3 items:")
                    for i, item in enumerate(data):
                        if i >= 3: break
                        print(f"  Item {i}: {type(item)}")
                        if isinstance(item, dict):
                            print(f"    Keys: {list(item.keys())}")
                            print(f"    'key' value: {item.get('key')}")
                            print(f"    'value' type: {type(item.get('value'))}")
                else:
                    print("File is not a list - showing first 100 chars:")
                    print(str(data)[:100])
                    
        except Exception as e:
            print(f"Error: {str(e)}")
            print("File content start:")
            with open(file_path, 'r') as file:
                print(file.read(200))  # Show first 200 chars

check_files()
