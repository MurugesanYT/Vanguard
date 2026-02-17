import torch
from ..models.multimodal_model import MultimodalModel
from ..data.dataset import ImageProcessor
import json

class InferenceEngine:
    def __init__(self, model_path, config, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.image_processor = ImageProcessor(image_size=config['vision_config']['image_size'])
        
        # Initialize model
        self.model = MultimodalModel(
            vocab_size=config['vocab_size'],
            vision_config=config['vision_config'],
            text_config=config['text_config'],
            num_functions=config['num_functions']
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Function registry
        self.functions = {}

    def register_function(self, name, func, schema):
        self.functions[name] = {
            "func": func,
            "schema": schema
        }

    def parse_function_call(self, text):
        # Simple parser for <function_call>name(args)</function_call>
        # In a real model, the model would generate this format
        import re
        match = re.search(r"<function_call>(.*?)\((.*?)\)</function_call>", text)
        if match:
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            # Very basic arg parsing, should be improved with JSON or AST
            args = {}
            if args_str:
                for arg in args_str.split(','):
                    k, v = arg.split('=')
                    args[k.strip()] = v.strip().strip('"').strip("'")
            return name, args
        return None, None

    @torch.no_grad()
    def chat(self, prompt, image_path=None, max_new_tokens=200):
        # 1. Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        image = None
        if image_path:
            image = self.image_processor(image_path).unsqueeze(0).to(self.device)
            
        # 2. Generate
        generated_ids = self.model.generate(
            input_ids, 
            images=image, 
            max_new_tokens=max_new_tokens
        )
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 3. Handle Function Calling
        # Check if the model generated a function call
        # (The generate method in MultimodalModel already has a trigger check)
        
        name, args = self.parse_function_call(response)
        if name and name in self.functions:
            print(f"Executing function: {name} with args: {args}")
            result = self.functions[name]["func"](**args)
            
            # 4. Continue generation with result
            new_prompt = f"{response}\n[Function Result: {result}]\n"
            new_inputs = self.tokenizer(new_prompt, return_tensors="pt").to(self.device)
            
            final_ids = self.model.generate(
                new_inputs["input_ids"],
                images=image,
                max_new_tokens=max_new_tokens
            )
            
            final_response = self.tokenizer.decode(final_ids[0], skip_special_tokens=True)
            return final_response
            
        return response
