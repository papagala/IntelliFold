## ⚙️ Installation
>To more complete installation instructions and usage, please refer below.

1. **Clone the repository**
    ```bash
    git clone https://github.com/IntelliGen-AI/IntFold.git
    cd IntFold
    ```

2. **Create and activate the environment(recommended)**
    ```bash
    conda env create -f environment.yaml
    conda activate intfold
    ```

3. **Install the package**
    - From PyPI (recommended):
      ```bash
      pip install intfold
      ```
    - From local wheel:
      ```bash
      pip install pypi/intfold-0.1.3-py3-none-any.whl
      ```
    - Editable install:
      ```bash
      pip install -e .
      ```

4. **(Optional) Download IntFold Cache Data Manually**<br>
    By default, model weights and CCD data are downloaded automatically(the directory is `~/.intfold`) when you run the inference. But you can also download by yourself.
    To download manually from [Our HuggingFace Repository](https://huggingface.co/intelligenAI/intfold):

    ```bash
    mkdir -p cache_data
    cd cache_data
    wget https://huggingface.co/intelligenAI/intfold/resolve/main/intfold_v0.1.0.pt
    wget https://huggingface.co/intelligenAI/intfold/resolve/main/ccd.pkl
    wget https://huggingface.co/intelligenAI/intfold/resolve/main/unique_protein_sequences.fasta 
    wget https://huggingface.co/intelligenAI/intfold/resolve/main/unique_nucleic_acid_sequences.fasta 
    wget https://huggingface.co/intelligenAI/intfold/resolve/main/protein_id_groups.json
    wget https://huggingface.co/intelligenAI/intfold/resolve/main/nucleic_acid_id_groups.json
    ```
    Your directory should look like:
    ```
    cache_data/
    ├── intfold_v0.1.0.pt
    ├── ccd.pkl
    ├── unique_protein_sequences.fasta
    ├── unique_nucleic_acid_sequences.fasta
    ├── protein_id_groups.json
    └── nucleic_acid_id_groups.json
    ```
    Place the downloaded files in the `cache_data/` directory before running inference.