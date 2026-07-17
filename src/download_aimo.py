import kagglehub

# Download latest version
path = kagglehub.dataset_download("taraashmittal/aimo-3")

print("Path to dataset files:", path)