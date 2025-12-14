# SemAI_Multi_agent_workflow

This repository is configured with **Git LFS (Large File Storage)** to support uploading large folders and files.

## Working with Large Files

### Prerequisites
Before cloning or working with this repository, ensure you have Git LFS installed:

```bash
# Install Git LFS (if not already installed)
# On Ubuntu/Debian
sudo apt-get install git-lfs

# On macOS
brew install git-lfs

# On Windows (using Git for Windows)
# Git LFS is included with Git for Windows 2.x

# Initialize Git LFS (one-time setup)
git lfs install
```

### Cloning the Repository
Simply clone the repository as usual. Git LFS will automatically handle large files:

```bash
git clone https://github.com/dibaotranworkcontact-cloud/SemAI_Multi_agent_workflow.git
```

### Adding Large Files
Large files matching the patterns in `.gitattributes` will automatically be tracked by Git LFS:

```bash
# Add your large files or folders
git add your-large-folder/
git commit -m "Add large files"
git push
```

### Supported Large File Types
This repository is configured to handle the following large file types via Git LFS:
- **Archives**: .zip, .tar, .tar.gz, .rar, .7z
- **Data files**: .csv, .parquet, .pkl, .h5, .hdf5
- **Machine Learning models**: .model, .weights, .pt, .pth, .ckpt, .pb, .onnx
- **Media files**: Images (.jpg, .png, .gif), Videos (.mp4, .avi, .mov), Audio (.mp3, .wav)
- **Documents**: .pdf, .docx, .pptx, .xlsx
- **And many more** (see `.gitattributes` for complete list)

### Tracking Additional File Types
To track additional file types with Git LFS:

```bash
git lfs track "*.your-extension"
git add .gitattributes
git commit -m "Track new file type with Git LFS"
```

### Benefits
- ✅ Upload large files and folders without hitting GitHub's file size limits
- ✅ Faster cloning and fetching for repositories with large files
- ✅ Same repository URL - no changes needed
- ✅ Seamless integration with your existing Git workflow

### Troubleshooting
If you encounter issues with large files:

```bash
# Check Git LFS status
git lfs status

# List files tracked by Git LFS
git lfs ls-files

# Verify Git LFS is installed
git lfs version
```

For more information, visit the [Git LFS documentation](https://git-lfs.github.com/).