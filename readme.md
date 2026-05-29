# Monte Carlo Evaluation of a Two-Qubit Variational Quantum Classifier with Zero-Noise Extrapolation for Intent Classification

This repository is optimized for **Jupyter Notebook** research with a focus on clean version control and metadata management. All collaborators **must** follow the setup instructions below to ensure repository integrity.

---

## 🛠 Prerequisites

Ensure you have the following installed on your system:

- **Python 3.14+**
- **Pipenv** (`pip install pipenv`)
- **Git**

---

## 🚀 Environment Setup

### 1. Initialize the Virtual Environment

Clone the repository and install the required dependencies (including development tools).

```bash
git clone https://github.com/Operator-Syn/csc133-research-paper.git && \
cd csc133-research-paper && \
pipenv install --dev -v
```

### 2. Configure Environment Variables

We use a `.env` file to configure the Python path so that `src/` is importable from any notebook without manual path hacks.

```bash
cp .env.example .env
```

The `.env.example` file is committed to the repository as a template. Your `.env` is local and will never be committed. It should contain:

```dotenv
PYTHONPATH=../src
```

> ℹ️ Pipenv loads `.env` automatically every time you enter `pipenv shell`, so `src/` will always be on the Python path for your Jupyter kernel.

### 3. Configure Notebook Tools

We use `nbdime` for clean diffs/merges and `pre-commit` to automatically strip notebook metadata (execution counts, etc.) before every commit.

```bash
# Activate the virtual environment
pipenv shell

# Enable nbdime for this repository
nbdime config-git --enable

# Install the pre-commit hooks
pre-commit install
```

### 4. Install the nbstripout Git Filter

This registers `nbstripout` as a **Git clean filter**, which automatically strips notebook output and metadata at `git add` time — before the commit hook even runs. This prevents the double-commit issue you'd otherwise hit when staging notebooks alongside other files.

```bash
nbstripout --install
```

> ⚠️ This writes to your local `.git/config` and is **not** committed to the repository. Every collaborator must run this **once** after cloning.

You only need to do this once per clone. After that, notebook outputs are stripped silently on every `git add` with no extra steps required.

---

## 🖥 Starting the Jupyter Server

Once your environment is active, launch the notebook server from the project root:

```bash
# If you're not already in the shell
pipenv shell

# Start the server
jupyter notebook
```

This will open Jupyter in your default browser at `http://localhost:8888`. From there you can create, edit, and run notebooks directly — no VS Code needed.

### Useful Launch Options

```bash
# Run on a specific port (if 8888 is taken)
jupyter notebook --port=8890

# Launch without opening a browser (e.g. on a remote machine)
jupyter notebook --no-browser

# Allow access from other devices on your network
jupyter notebook --ip=0.0.0.0
```

> ⚠️ **Heads up:** If you use `--ip=0.0.0.0`, anyone on your local network can access the server. Only use this in a trusted environment.

---

## 📋 Collaboration Workflow

### Dealing with Jupyter Notebooks

Because `.ipynb` files are JSON, standard Git tools often struggle with them. We have configured specialized utilities to handle them correctly:

- **Clean Diffs:** When you run `git diff`, you will see code changes only, without the JSON metadata noise.
- **Smart Merges:** If a notebook has a conflict, **do not** fix it manually in a text editor. Instead, use the visual merge tool:
  ```bash
  pipenv run nbdime mergetool
  ```
  This launches a web-based 3-pane merge editor in your browser.
- **Automatic Scrubbing:** The `nbstripout` Git filter strips outputs and metadata automatically on `git add`. The `pre-commit` hook provides a second pass as a safety net at commit time.

### Working on Features

1. **Create a branch:** `git checkout -b feature/your-feature-name`
2. **Commit your changes:** If a commit fails on the first try, it is likely because `pre-commit` cleaned your notebook metadata. Simply `git add` the file again and re-commit.

---

## 📂 Repository Structure

| File | Purpose |
|------|---------|
| `.env.example` | Template for local environment variables — copy to `.env` after cloning |
| `.pre-commit-config.yaml` | Configuration for notebook scrubbing |
| `.gitattributes` | Settings for `nbdime` diff/merge drivers |
| `Pipfile` | Dependency management |
| `.gitignore` | Configured to ignore `.env`, `.orig` merge backups, and local checkpoints |

---

## 💡 Tips

> **Adding new packages?** Always use `pipenv install <package>` instead of `pip install` so the `Pipfile` stays updated for everyone on the team.

---

## 📜 License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<!-- > **XQubit Project** — Jupyter Notebook research with clean version control and metadata management. -->

Copyright (C) 2026 **Operator-Syn**

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **without any warranty** — without even the implied warranty of merchantability or fitness for a particular purpose. See the full [GNU General Public License](https://www.gnu.org/licenses/) for more details.