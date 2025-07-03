### Run with Command line

If you set up `IntFold` by `pip`, you can run the following command to do model inference:

```bash

# run with example yaml, which contains precomputed msa files(.a3m or .csv). the default seed is 42.
## a3m MSA file type
intfold predict ./examples/5S8I_A.yaml --out_dir ./output 
## csv MSA file type
intfold predict ./examples/7yds.yaml --out_dir ./output 

# Predict with a directory of YAMLs
intfold predict ./examples --out_dir ./output

# run with 5 seeds(mutiple seeds are splited them by comma) and 5 samples (the default parameters for AlphaFold3).
intfold predict ./examples/5S8I_A.yaml --seed 42,43,44,45,46 --num_diffusion_samples 5 --out_dir ./output

# if the input yaml file do not contain precomputed msa paths, you can set --use_msa_server to search msa(need internet connection and would take some time) 
# and use greedy msa pairing strategy
# and then predict
intfold predict examples/examples_wo_msa/example_without_msa.yaml --out_dir ./output --seed 42,66 --use_msa_server --msa_pairing_strategy greedy

# only run the data processing step, and not run the model.
intfold predict ./examples/5S8I_A.yaml --out_dir ./output --only_run_data_process

```

### Run with Bash Script

The aurguments is the same as `intfold predict`, and you can set the parameters in the script.
you can get the help information by running `intfold predict --help` or `python run_intfold.py --help`
```bash
bash predict.sh
## or
## python run_intfold.py ....
```
Common arguments of this `scripts`/`intfold predict` are explained as follows:
* `--out_dir` (`PATH`, default: `./`)  
  The path where to save the predictions.
* `--cache` (`PATH`, default: `~/.intfold`)  
  The directory where to download the data and model. Will use environment variable `INTFOLD_CACHE` as an absolute path if set.
* `--num_workers` (`INTEGER`, default: `4`)  
  The number of dataloader workers to use for prediction.
* `--precision` (`str`, default: `bf16`)  
  Sets precision, lower precision improves runtime.
* `--seed` (`INTEGER`, default: `42`)  
  Random seed (single int or multiple ints separated by comma, e.g., '42' or '42,43').
* `--recycling_iters` (`INTEGER`, default: `10`)  
  Number of recycling iterations.
* `--num_diffusion_samples` (`INTEGER`, default: `5`)  
  The number of diffusion samples.
* `--sampling_steps` (`INTEGER`, default: `200`)  
  The number of diffusion sampling steps to use.
* `--output_format` (`[pdb,mmcif]`, default: `mmcif`)  
  The output format to use for the predictions (pdb or mmcif).
* `--override` (`FLAG`, default: `False`)  
  Whether to override existing found predictions.
* `--use_msa_server` (`FLAG`, default: `False`)  
  Whether to use the MMSeqs2 server for MSA generation.
* `--msa_server_url` (`str`, default: `https://api.colabfold.com`)  
  MSA server url. Used only if `--use_msa_server` is set.
* `--msa_pairing_strategy` (`str`, default: `complete`)  
  Pairing strategy to use. Used only if `--use_msa_server` is set. Options are 'greedy' and 'complete'.
* `--no_pairing` (`FLAG`, default: `False`)  
  Whether to use pairing for Protein Multimer MSA generation.
* `--only_run_data_process` (`FLAG`, default: `False`)  
  Whether to only run data processing, and not run the model.
* `--return_similar_seq` (`FLAG`, default: `False`)
  Whether to return sequences similar to those in the training PDB dataset during inference. You can use these similar sequences and its PDB ids to do further analysis, such as a reference structure.
  > Before using this option, please make sure the mmseqs2 tool is installed, you can install it by running `conda install -c conda-forge -c bioconda mmseqs2`