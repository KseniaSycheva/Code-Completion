# Code-Completion
Code completion assignment code. 

## Setup 
```sh
poetry install
``` 

## Prepare Dataset
To prepare code completion dataset on your examples run
```sh
python prepare_data.py --project-root <path-to-your-project> --hf-save-path <your-hf-path> --hf-token <your-hf-token>
``` 

Other arguments: 
- n-examples: desired number of examples in the dataset
- n-lines-prefix-suffix: number of lines to include in prefix and suffix 
- print-n-examples: number of examples to print; can be useful for debugging

Dataset is created for single-line code completion. 

## Inference
To generate completions for the dataset run
```sh
python inference.py --model-name <model-name-on-hf> --dataset-name <dataset-name-on-hf>
``` 

Other arguments: 
- n-examples: number of examples to use in evaluation
- quantize: whether to load model in 8-bit (when used in colab, recommended for models with # params >= 7b)

Completion examples for CodeLlama-7b-Instruct-hf are available [here](examples.txt). 
To replicate results, it is recommended to run [notebook](code_completion.ipynb) in colab. 

## Evaluation
The following metrics are used: 
- Exact match: the rate at which the input predicted strings exactly match their references.
- Chrf: uses the F-score statistic for character n-gram matches.
- BLEU: computes n-grams 'clipped' precision + penalizes for brevity. 
- ROUGE: measures token overlap with focus on recall. Has many limitations, since originally designed for summarization.
- BertScore: embedding-based metric; correlated with my judgement of the completions, as it recognizes not just surface-level matches but also deeper similarities in meaning.