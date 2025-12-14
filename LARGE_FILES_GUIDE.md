# Quick Start Guide for Large File Uploads

## Overview
This repository uses **Git LFS (Large File Storage)** to handle large files and folders efficiently. Git LFS replaces large files with text pointers inside Git, while storing the file contents on a remote server.

## One-Time Setup (For New Users)

1. **Install Git LFS** (if not already installed):
   ```bash
   # macOS
   brew install git-lfs
   
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # Windows - included with Git for Windows 2.x
   ```

2. **Initialize Git LFS** (one-time per user):
   ```bash
   git lfs install
   ```

## Cloning This Repository

Clone normally - Git LFS handles everything automatically:
```bash
git clone https://github.com/dibaotranworkcontact-cloud/SemAI_Multi_agent_workflow.git
```

## Adding Large Files/Folders

Simply add and commit as usual - files matching the configured patterns will be automatically handled by LFS:

```bash
# Add a large folder
git add my-large-folder/

# Commit
git commit -m "Add large dataset"

# Push
git push
```

## What File Types Are Supported?

All common large file types are pre-configured (see `.gitattributes`):
- Archives: `.zip`, `.tar`, `.gz`, `.rar`, `.7z`
- Data: `.csv`, `.parquet`, `.pkl`, `.h5`, `.hdf5`
- ML Models: `.pt`, `.pth`, `.ckpt`, `.pb`, `.onnx`, `.model`
- Media: Images, videos, audio files
- Documents: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- And many more...

## Adding New File Types

To track additional file types with LFS:
```bash
git lfs track "*.your-extension"
git add .gitattributes
git commit -m "Track new file type"
```

## Checking LFS Status

```bash
# See which files are tracked by LFS
git lfs ls-files

# Check LFS status
git lfs status

# See tracked patterns
git lfs track
```

## Benefits

✅ **No repository URL changes** - keep using the same remote URL  
✅ **Upload large files** without hitting GitHub's 100MB file size limit  
✅ **Faster operations** - cloning and fetching are faster  
✅ **Transparent** - works seamlessly with your normal Git workflow  

## Troubleshooting

**Problem**: Git LFS not installed  
**Solution**: Install Git LFS using the commands above

**Problem**: Large file rejected during push  
**Solution**: Make sure the file type is listed in `.gitattributes`. If not, add it with `git lfs track "*.extension"`

**Problem**: Want to see actual file size  
**Solution**: Use `git lfs ls-files -s` to see sizes

## Learn More

- [Git LFS Official Site](https://git-lfs.github.com/)
- [GitHub's Git LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
