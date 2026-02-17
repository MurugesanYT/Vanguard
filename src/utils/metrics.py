import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.rouge_score import Rouge

class Evaluator:
    def __init__(self):
        self.rouge = Rouge()

    def compute_bleu(self, reference, candidate):
        """Compute BLEU-4 score."""
        ref_tokens = [reference.split()]
        cand_tokens = candidate.split()
        return sentence_bleu(ref_tokens, cand_tokens)

    def compute_rouge(self, reference, candidate):
        """Compute ROUGE scores."""
        if not reference or not candidate:
            return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        scores = self.rouge.get_scores(candidate, reference)[0]
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }

    def compute_function_accuracy(self, predictions, targets):
        """Compute accuracy for function calling."""
        correct = 0
        for pred, target in zip(predictions, targets):
            if pred['name'] == target['name'] and pred['args'] == target['args']:
                correct += 1
        return correct / len(targets) if targets else 0
