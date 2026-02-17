import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageProcessor:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            # Return a zero tensor if image loading fails
            return torch.zeros(3, 224, 224)

class LLaVADataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, image_processor, max_length=512):
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = []
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # LLaVA format: conversations is a list of dicts
        # We'll take the first human-gpt pair for simplicity in this version
        convs = item['conversations']
        human_text = ""
        gpt_text = ""
        
        for msg in convs:
            if msg['from'] == 'human':
                human_text = msg['value'].replace('<image>\n', '').replace('\n<image>', '')
            elif msg['from'] == 'gpt':
                gpt_text = msg['value']
                break # Just take the first pair
        
        full_text = f"User: {human_text}\nAssistant: {gpt_text}"
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        # Image processing
        image = None
        if 'image' in item:
            image_path = os.path.join(self.image_folder, item['image'])
            image = self.image_processor(image_path)
        else:
            image = torch.zeros(3, 224, 224)
            
        return {
            'input_ids': input_ids,
            'images': image,
            'targets': input_ids.clone(),
            'call_label': torch.tensor(0), # LLaVA doesn't have function calls
            'func_label': torch.tensor(0)
        }

class ToolBenchDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, max_length=512):
        # This would load the ToolBench data
        # For now, we'll implement a placeholder that can be expanded
        self.data = [] # Load your toolbench json here
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Implementation for ToolBench format
        pass
