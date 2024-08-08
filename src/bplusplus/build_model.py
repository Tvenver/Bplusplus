import os

#TODO implement with tempfile


# def build_model(scientific_names: list[str], images_per_species: int) -> str:
    
#     collect_images(
#         scientific_names=scientific_names, 
#         images_per_species=images_per_species
#     )




# import tempfile
# import os
# import shutil

# temp_dir = None

# try:
#     # Create a temporary directory
#     temp_dir = tempfile.mkdtemp()
    
#     # Use the temporary directory
#     file_path = os.path.join(temp_dir, "example.txt")
    
#     with open(file_path, "w") as f:
#         f.write("Hello, temporary world!")
    
#     # Read the file content
#     with open(file_path, "r") as f:
#         content = f.read()
#         print(f"File content: {content}")
    
#     # Do more work...
#     print(f"Temporary directory path: {temp_dir}")

# finally:
#     # Clean up the temporary directory
#     if temp_dir and os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)
#         print(f"Cleaned up temporary directory: {temp_dir}")
