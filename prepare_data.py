import argparse
import os

from datasets import Dataset
from pyarrow import Table


def create_completion_examples_single_file(
        file_path: str,
        dataset: dict[str, list[str]],
        n_lines_prefix_suffix: int
) -> dict[str, list[str]]:

    with open(file_path) as fp:
        content = fp.readlines()

    content = list(filter(lambda x: len(x) > 5, content))

    if len(content) % 2 == 0:
        content = content[:-1]

    skipped_lines_indices = [i for i in range(n_lines_prefix_suffix, len(content), n_lines_prefix_suffix + 1)]

    for i in skipped_lines_indices:
        dataset["prefix"].append("".join(content[i-n_lines_prefix_suffix:i]))
        dataset["suffix"].append("".join(content[i+1:i+1+n_lines_prefix_suffix]))
        dataset["reference"].append(content[i])

    return dataset


def create_completion_examples_directory(
        path_to_dir: str,
        n_examples: int,
        n_lines_prefix_suffix: int
) -> Dataset:
    dataset: dict[str, list[str]] = {"prefix": [], "suffix": [], "reference": []}

    for dir_name, _, file_names in os.walk(path_to_dir):
        for file_name in file_names:
            if file_name.endswith(".py"):
                abs_filename = os.path.join(dir_name, file_name)
                dataset = create_completion_examples_single_file(abs_filename, dataset, n_lines_prefix_suffix)

    table = Table.from_pydict(dataset)
    return Dataset(table[:n_examples])


def visualize_examples(dataset: Dataset, n_examples: int):
    for i in range(n_examples):
        print(dataset[i]["prefix"])
        print("===="*5)
        print(dataset[i]["reference"])
        print("====" * 5)
        print(dataset[i]["suffix"])
        print("\n")
    print(len(dataset))


def main(args: argparse.Namespace):
    dataset = create_completion_examples_directory(args.project_root, args.n_examples, args.n_lines_prefix_suffix)
    visualize_examples(dataset, args.print_n_examples)
    dataset.push_to_hub(args.hf_save_path, token=args.hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, required=True,
                        help="Path to root of the folder to parse.")
    parser.add_argument("--n-examples", type=int, default=50, help="Number of examples to collect.")
    parser.add_argument("--n-lines-prefix-suffix", type=int, default=5,
                        help="Number of lines used in prefix and suffix.")
    parser.add_argument("--hf-save-path", type=str, required=True, help="Huggingface save path.")
    parser.add_argument("--hf-token", type=str, required=True, help="Huggingface token")
    parser.add_argument("--print-n-examples", type=int, default=10,
                        help="Number of examples to print from dataset.")

    parsed_args = parser.parse_args()
    main(parsed_args)
