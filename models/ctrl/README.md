## Control Tokens

To train a [CTRL](https://aclanthology.org/2021.acl-long.58/) model, you need to provide control tokens for each utterance in training and validation data.


### Requirements
Computing control tokens requires additional dependencies:

```bash
pip install -r models/ctrl/ctrl_requirements.txt
```

### Run
To determine control tokens for a given dataset, run the following command:

```bash
python models/ctrl/ctrl_data.py --split train --output_dir /path/to/output
``` 

Set `CUDA_VISIBLE_DEVICES` if you wish to run the command to GPU(s).

Available arguments for the above command are:
- `--input_file` (optional): Path to the input file. If not provided, data will be loaded from Huggingface's datasets.
- `--output_dir`: Path to the output directory. Optional if `--input_file` refers to a file.
- `--split`: Split of the data to use. One of `train` or `valid`.
- `--nli_model`: NLI model for predicting entailment labels.
- `--per_device_batch_size`: Batch size per device.
- `--max_length`: Maximum sequence length.


