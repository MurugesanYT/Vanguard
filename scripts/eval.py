import torch
from src.inference.engine import InferenceEngine
from src.utils.metrics import Evaluator
from transformers import AutoTokenizer
import yaml
import json

def evaluate_model(config_path, test_data_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<function_call>', '</function_call>']})
    
    engine = InferenceEngine(
        model_path=config['model_checkpoint'],
        config=config,
        tokenizer=tokenizer
    )
    
    evaluator = Evaluator()
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        
    results = {
        "bleu": [],
        "rouge-l": [],
        "func_acc": []
    }
    
    for item in test_data:
        prompt = item['prompt']
        image_path = item.get('image_path')
        reference = item['reference']
        
        response = engine.chat(prompt, image_path=image_path)
        
        # Text metrics
        results['bleu'].append(evaluator.compute_bleu(reference, response))
        results['rouge-l'].append(evaluator.compute_rouge(reference, response)['rouge-l'])
        
        # Function calling metrics (if applicable)
        if 'target_func' in item:
            pred_name, pred_args = engine.parse_function_call(response)
            is_correct = (pred_name == item['target_func']['name'] and 
                          pred_args == item['target_func']['args'])
            results['func_acc'].append(float(is_correct))
            
    # Aggregate
    summary = {k: sum(v)/len(v) if v else 0 for k, v in results.items()}
    print("Evaluation Summary:")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    # Example usage
    # evaluate_model("configs/default_config.yaml", "data/test_data.json")
    pass
