import os

def list_files(folder):
    for root, dirs, files in os.walk(folder):
        # Print the current directory
        print(root)
        
        # Print all files in the current directory
        for file in files:
            print(os.path.join(root, file))
        
        # Print all subdirectories in the current directory
        for dir in dirs:
            print(os.path.join(root, dir))

# Example usage
folder_path = 'C:\\Users\\sisun\\Documents\\Blog\\HugoBookBlog\\mydocs\\content'

list_files(folder_path)