import os

def list_project_files(root_dir="."):
    print(f"Scanning project in: {os.path.abspath(root_dir)}\n")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden/system directories
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        
        if filenames:
            print(f"[{os.path.relpath(dirpath, root_dir)}]")
            for filename in filenames:
                if not filename.startswith('.'):
                    print(f"  └── {filename}")
            print()

if __name__ == "__main__":
    # Change this to your project root if not running from there
    list_project_files(root_dir=".")
