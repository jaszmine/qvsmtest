## EXACT installation steps

### 1. Open VS Code and Launch Terminal
First, open VS Code and open a terminal:

- Windows/Linux: Ctrl + `` (backtick)
- Mac: Cmd + `` (backtick)

### 2. Clone the Repository
```bash
git clone https://github.com/jaszmine/qvsmtest.git
```

### 3. Create and Activate Virtual Environment
On Mac/Linux:
```bash
# Create virtual environment
python3 -m venv gene_analysis_env

# Activate environment
source gene_analysis_env/bin/activate
```

On Windows:
```bash
# Create virtual environment
python -m venv gene_analysis_env

# Activate environment
gene_analysis_env\Scripts\activate
```
You should see (gene_analysis_env) at the beginning of your command line

### 4. Install Requirements
Create a requirements.txt file in your project folder with the content above, then run:

```bash
pip3 install -r requirements.txt
```

### 5. Run Script
In the terminal run:
```bash
python3 qvsm9-0.py
```