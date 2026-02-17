from src.models.multimodal_model import MultimodalModel
from src.training.trainer import MultimodalTrainer
from src.data.dataset import LLaVADataset, ImageProcessor
from transformers import AutoTokenizer
import yaml
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    # Add special tokens for function calling
    tokenizer.add_special_tokens({'additional_special_tokens': ['<function_call>', '</function_call>']})
    # Ensure pad token is set for DistilGPT2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    if args.mode == "train":
        # Initialize model
        model = MultimodalModel(
            vocab_size=vocab_size,
            vision_config=config['vision_config'],
            text_config=config['text_config'],
            num_functions=config['num_functions'],
            use_pretrained=config.get('use_pretrained', False)
        )
        
        # Real data loading
        image_processor = ImageProcessor(image_size=config['vision_config']['image_size'])
        
        # Path to LLaVA data
        llava_json = os.path.join("data", "llava_instruct", "llava_instruct_150k.json")
        image_folder = config.get('image_folder', 'data/coco/train2017') # Default COCO path
        
        if not os.path.exists(llava_json):
            print(f"⚠️ Warning: {llava_json} not found. Using mock data for now.")
            # Fallback to mock data if file doesn't exist
            mock_data = [{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}]
            dataset = LLaVADataset(None, None, tokenizer, image_processor)
            dataset.data = mock_data
        else:
            dataset = LLaVADataset(
                json_path=llava_json,
                image_folder=image_folder,
                tokenizer=tokenizer,
                image_processor=image_processor
            )
        
        trainer = MultimodalTrainer(
            model=model,
            train_dataset=dataset,
            val_dataset=dataset,
            config=config['training_config']
        )
        
        trainer.train()
    
    elif args.mode == "infer":
        from src.inference.engine import InferenceEngine
        
        engine = InferenceEngine(
            model_path=config['model_checkpoint'],
            config=config,
            tokenizer=tokenizer
        )
        
        # Example function registration
        def get_weather(location, units="celsius"):
            return f"The weather in {location} is 20 degrees {units}."
            
        engine.register_function(
            "get_weather", 
            get_weather, 
            {"location": "string", "units": "string"}
        )
        
        response = engine.chat("What is the weather in Paris?")
        print(f"Model response: {response}")

if __name__ == "__main__":
    main()
