import argparse
from pprint import pprint

import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel

from metrics import compute_metrics


def load_model(model_name: str, quantize: bool = False) \
        -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Loads hf model and corresponding tokenizer.

    When working with large models, it is advised to quantize it.
    When running in colab (free) 7b models do not fit without quantization.

    :param model_name: path to model on hf model hub.
    :param quantize: if true, model is loaded in 8 bit.
    :return: tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantize:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model


def get_llama_code_completion_prompt(prefix: str, suffix: str,
                                     tokenizer: PreTrainedTokenizer) -> list[list[int]]:
    """Prepares Code Llama prompt.

    Code Llama paper: https://arxiv.org/pdf/2308.12950.
    Prompt format for code infilling is described in https://arxiv.org/pdf/2207.14255 (Appendix D).

    :param prefix: lines of code preceding masked line.
    :param suffix: lines of code following masked line.
    :param tokenizer: tokenizer.
    :return: list of tokens.
    """
    prefix_tokens = tokenizer(prefix)["input_ids"][1:]
    suffix_tokens = tokenizer(suffix)["input_ids"][1:]

    pre_suf = tokenizer("<PRE><SUF>")["input_ids"]
    mid = tokenizer("<MID>")["input_ids"][1:]

    # wrap it in another list, since models work with batched inputs
    return [pre_suf + suffix_tokens + mid + prefix_tokens]


def extract_single_line(predicted: str, prefix: str) -> str:
    """Extract first line from predicted code block.

    Code Llama predicts multiple lines. Since I consider single-line prediction here,
    the first line is extracted from prediction.

    :param predicted: output of the model (multi-line completion).
    :param prefix: lines of code preceding masked line.
    :return: first predicted line.
    """
    mid_ind = predicted.find("<MID>")
    predicted = predicted[mid_ind:]
    processed = predicted.lstrip("<MID>").strip()
    return processed[len(prefix):].strip().splitlines()[0]


def generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, entry: dict[str, str]) -> str:
    """Generates a single-line completion.

    :param model: model being evaluated.
    :param tokenizer: tokenizer corresponding to the model.
    :param entry: an entry from dataset.
    :return: single line completion for a given entry.
    """
    tokens = get_llama_code_completion_prompt(entry["prefix"], entry["suffix"], tokenizer)
    print("PROMPT:", tokenizer.decode(tokens[0]))
    print("====" * 5)

    tokens = torch.tensor(tokens).to(model.device)

    # generation parameters are such that outputs of the model are not deterministic
    # to make model deterministic, use do_sample=False and optionally beam search
    predicted = model.generate(tokens, do_sample=True, top_k=50, top_p=0.9, max_new_tokens=100).squeeze(0)
    decoded = tokenizer.decode(predicted)
    single_line = extract_single_line(decoded, entry["prefix"])

    print("REFERENCE:", entry["reference"])
    print("====" * 5)
    print("PREDICTED:", single_line)
    print("\n")

    return single_line


def main(args: argparse.Namespace):
    tokenizer, model = load_model(args.model_name, args.quantize)
    dataset = load_dataset(args.dataset_name)

    references: list[str] = []
    predictions: list[str] = []

    for row in dataset["train"].select(range(args.n_examples)):
        prediction = generate(model, tokenizer, row)
        predictions.append(prediction)
        references.append(row["reference"])

    # compute metrics
    metrics = compute_metrics(predictions, references)
    pprint(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True,
                        help="Path to local model or model on hf.")
    parser.add_argument("--quantize", action="store_true", help="Whether to load quantized model.")
    parser.add_argument("--dataset-name", type=str, help="Path to local dataset or dataset on hf.")
    parser.add_argument("--n-examples", type=int, default=10,
                        help="Number of examples to generate.")

    parsed_args = parser.parse_args()
    main(parsed_args)
