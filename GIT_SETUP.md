# GitHub Setup (SLAI Project)

Use this once to connect this folder to your GitHub repo.

## 1) Initialize git in this folder

```powershell
cd D:\SLAIv0.1
git init
```

## 2) Set your Git identity (replace values)

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 3) Create first commit

```powershell
git add .
git commit -m "Initial SLAI codebase"
```

## 4) Create an empty GitHub repository

Create a new repo on GitHub (do not add README from GitHub UI).

## 5) Connect local repo to GitHub (replace URL)

```powershell
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

## Optional: use GitHub CLI

If `gh` is installed and authenticated:

```powershell
gh repo create <your-repo> --private --source . --remote origin --push
```

## Notes

- `.gitignore` already excludes checkpoints, logs, runtime memory, and build outputs.
- If you want to version large models later, use Git LFS instead of normal git.
