import argparse
from pprint import pprint

import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel

from metrics import compute_metrics


def load_model(model_name: str, quantize: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantize:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model


def get_llama_code_completion_prompt(prefix: str, suffix: str, tokenizer: PreTrainedTokenizer):
    prefix_tokens = tokenizer(prefix)["input_ids"][1:]
    suffix_tokens = tokenizer(suffix)["input_ids"][1:]

    pre_suf = tokenizer("<PRE><SUF>")["input_ids"]
    mid = tokenizer("<MID>")["input_ids"][1:]
    return [pre_suf + suffix_tokens + mid + prefix_tokens]


def extract_single_line(predicted: str, prefix: str):
    mid_ind = predicted.find("<MID>")
    predicted = predicted[mid_ind:]
    processed = predicted.lstrip("<MID>").strip()
    return processed[len(prefix):].strip().splitlines()[0]


def generate(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, entry: dict[str, str]):
    tokens = get_llama_code_completion_prompt(entry["prefix"], entry["suffix"], tokenizer)
    print("PROMPT:", tokenizer.decode(tokens[0]))
    print("====" * 5)

    tokens = torch.tensor(tokens).to(model.device)

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
