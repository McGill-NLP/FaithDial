# FaithDial: A Faithful Benchmark for Information-Seeking Dialogue

This repository hosts the code and pre-trained models for our paper [FaithDial: A Faithful Benchmark for Information-Seeking Dialogue](https://arxiv.org/pdf/2204.10757.pdf).
Also, it hosts the data annotations for our NAACL paper [On the origin of hallucination in dialogue systems](https://arxiv.org/pdf/2204.07931.pdf).
For more information, please visit the [project page](https://mcgill-nlp.github.io/FaithDial/).

<!-- Thanks for your interest in our repo! -->
<!-- We were inspired by SimCSE to organize this repo! 🖖 -->

**************************** **Updates** ****************************
* 4/25: We released the [FaithDial paper](https://arxiv.org/abs/2204.10757) and launched the [project page](https://mcgill-nlp.github.io/FaithDial/). Check them out!
* 4/15: We released [our paper](https://arxiv.org/abs/2204.07931), to appear at NAACL 2022!

## Quick Links

  - [Overview](#overview)
  - [Data](#data)
  - [Use with Huggingface (coming soon!)](#use-with-huggingface)
  - [Train Your Models](#train-your-models)
    - [Requirements](#requirements)
    - [Data Format](#data-format)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Generation](#generation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)


## Overview
The goal of information-seeking dialogue is to respond to user queries with natural language utterances that are grounded on knowledge sources.
Dialogue systems, however, often hallucinate, i.e. generate unsupported utterances, as they amplify the noise found in existing training datasets.
To mitigate this behavior, we adopt a data-centric solution and create FaithDial, a new benchmark for hallucination-free dialogues. Annotators were asked to edit the hallucinated utterances in a pre-existing dataset to ensure they are faithful to knowledge sources and re-purpose the role of the interlocutor from a human wizard to a domain-expert bot.

## Data
The dataset is hosted on [Huggingface's datasets](https://github.com/huggingface/datasets):

```python
from datasets import load_dataset

dataset = load_dataset("McGill-NLP/FaithDial")
```

## Use with Huggingface
We'll release our fine-tuned models soon! Stay tuned!

## Train Your Models
The code for all the models in the paper is available in [models](models/), which can be used to reproduce our results or to train your own models.

### Requirements
First, install Pytorch 1.7+ from the [official website](https://pytorch.org) and then, clone this repository and install the dependencies:

```bash
git clone git@github.com:McGill-NLP/FaithDial.git
pip install -r requirements.txt
```

Our code is tested with Python `3.8`, and Pytorch `1.7.1` with CUDA `11.0`.

### Data Format
By default, our code loads data from the Huggingface's datasets. But, you can also provide your own data with the following format:

```text
[
  {
    "utterances": [
      ... // prior utterances, 
      {
        "history": [
          "Have you ever been to a concert? They're so fun!",
          "No I cannot as a bot. However, have you been to Madonna's? Her 10th concert was used to help her 13th album called \"Rebel Heart\".",
          "Yeah I've heard of it but never went or what it was for. Can you tell me more about it?"
        ],
        "speaker": "Wizard",
        "knowledge": "It began on September 9, 2015, in Montreal, Canada, at the Bell Centre and concluded on March 20, 2016, in Sydney, Australia at Allphones Arena.",
        "original_response": "It started in September of 2015 and ran all the way through March of 2016. Can you imagine being on the road that long?",
        "response": "Sure. The concert started in September 9th of 2015 at Montreal, Canada. It continued till 20th of March of 2016, where it ended at Sydney, Australia.",
        "BEGIN": [
          "Hallucination",
          "Entailment"
        ],
        "VRM": [
          "Disclosure",
          "Question"
        ]
      }, 
      ... // more utterances
    ]
  }, 
  ... // more dialogues
]
```
In the above example, `original_response`, `BEGIN`, and `VRM` are optional and don't have to be provided for your own data.

### Training
Here is how to train a model:

```bash
python models/dialog.py --model_name_or_path t5-base \ 
  --do_train \
  --output_dir /path/to/output_dir \
  --fp16 \
  --train_batch_size 16 \
  --num_train_epochs 10 \
  --warmup_ratio 0.04 \
  --max_seq_length 512
```

To run on multiple GPUs, set `CUDA_VISIBLE_DEVICES`. By default, training early stops and the best model is saved at `/path/to/output_dir/best_model`.

Other arguments for training are as follows:
- `--learning_rate`: Initial learning rate for Adam.
- `--gradient_accumulation_steps`: Number of steps to accumulate gradient before performing a backward/update pass.
- `--enable_infonce`: Whether to use the InfoNCE model. Note that `negative_samples` must be present in the input data for contrastive learning. Also, `--fp16` should not be set.
- `--max_negative_samples`: The number of negative samples per training example (Works only when InfoNCE is enabled).
- `--inbatch_negatives`: Whether to use inbatch negative sampling (Works only when InfoNCE is enabled).
- `--loss_truncation`: Whether to use [loss truncation](https://aclanthology.org/2020.acl-main.66/).
- `--ctrl`: Whether to use [controlled generation](https://aclanthology.org/2021.acl-long.58/). Note that `control_tokens` must be present in the input data. To learn about how to compute control tokens, see [here](models/ctrl/). 
- `--train_dataset_path` (optional): Path to your own training dataset.
- `--eval_dataset_path` (optional): Path to your own validation dataset.

For a complete list of arguments, take a look at [models/dialog.py](models/dialog.py#L180) and [models/lightning_base.py](models/lightning_base.py#L268).


### Evaluation
To compute perplexity of a model on the validation data, simply run:

```bash
python models/dialog.py --model_name_or_path /path/to/model/best_model \
  --do_eval \
  --eval_batch_size 16
```

For the test data, `--do_eval` should be replaced with `--do_test`.
Note that evaluation should be run on a single GPU.

To compute other metrics (BLEU, ROUGE, F1, BERTScore, and Q^2), reported in the paper, we used the scripts, provided in [https://github.com/orhonovich/q-squared](https://github.com/orhonovich/q-squared).

### Generation
To generate a response, simply run:

```bash
python models/generate.py --model_name_or_path /path/to/model/best_model --do_sample --top_p 0.6
```
Arguments for generation are as follows:
- `--output` (optional): Path of the output directory to save the generated responses.
- `--dataset_path` (optional): Path to your own dataset.
- `--control_tokens` (optional): Control tokens, prepended to the sequence, for controlled generation.
- `--max_length` (default: 100): Maximum length of the generated sequence.

For a complete list of arguments, refer to [models/generate.py](models/generate.py#L97).

## Bugs or questions?

If you have any questions (:question:) related to the code, or encounter any problems (:hammer_and_wrench:), or want to report a bug (:bug:), feel free to open an issue.

## Citation

If you want to cite our papers, please use:

```bibtex
@article{dziri2022faithdial,
  title={FaithDial: A Faithful Benchmark for Information-Seeking Dialogue},
  author={Dziri, Nouha and Kamalloo, Ehsan and Milton, Sivan and Zaiane, Osmar and Yu, Mo and Ponti, Edoardo and Reddy, Siva},
  journal={arXiv preprint, arXiv:2204.10757},
  year={2022},
  url={https://arxiv.org/abs/2204.10757}
}
```

and

```bibtex
@article{dziri2022origin,
  title={On the origin of hallucination in dialogue systems},
  author={Dziri, Nouha and Milton, Sivan and Yu, Mo and Zaiane, Osmar and Reddy, Siva},
  journal={arXiv preprint, arXiv:2204.07931},
  year={2022},
  url={https://arxiv.org/abs/2204.07931}
}
```

## License

This work is licensed under the MIT license. See [LICENSE](LICENSE) for details.