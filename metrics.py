import sacrebleu
from evaluate import load


bleu = load("bleu")
rouge = load("rouge")
exact_match = load("exact_match")
bert_score = load("bertscore")


def compute_bleu(prediction: str, reference: str):
    return bleu.compute(predictions=[prediction], references=[reference])["bleu"]


def compute_rouge(prediction: str, reference: str):
    return rouge.compute(predictions=[prediction], references=[reference])["rougeLsum"]


def compute_exact_match(prediction: str, reference: str):
    return exact_match.compute(predictions=[prediction], references=[reference])["exact_match"]


def compute_bertscore(prediction: str, reference: str):
    return bert_score.compute(predictions=[prediction], references=[reference], lang="en")["f1"][0]


def compute_chrf(prediction, reference):
    chrf_score = sacrebleu.corpus_chrf([prediction], [reference])
    return chrf_score.score


metrics_to_function = {
    "exact_match": compute_exact_match,
    "chrf": compute_chrf,
    "bleu": compute_bleu,
    "rouge": compute_rouge,
    "bertscore": compute_bertscore,
}


def compute_metrics(predictions: list[str], references: list[str]):
    assert len(predictions) == len(references)

    metrics: dict[str, list[float]] = {k: [] for k in list(metrics_to_function.keys())}
    for i in range(len(predictions)):
        for metric_name, metric in metrics_to_function.items():
            metrics[metric_name].append(metric(predictions[i], references[i]))

    avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    return avg_metrics
