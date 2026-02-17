import torch
import torch.nn as nn
import torch.nn.functional as F
from .vision_encoder import VisionTransformer
from .decoder import TextDecoder, precompute_freqs_cis, apply_rotary_emb
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from typing import Optional, Dict, Any

class MultimodalProjector(nn.Module):
    def __init__(self, vision_dim, text_dim):
        super().__init__()
        # LLaVA-1.5 style projector: 2-layer MLP with GELU
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, x):
        return self.projector(x)

class FunctionCallingHead(nn.Module):
    def __init__(self, embed_dim, num_functions=100):
        super().__init__()
        # Improved head with multiple layers for better classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2) # [No Call, Call]
        )
        self.function_selector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_functions)
        )
        
    def forward(self, x):
        call_logits = self.classifier(x)
        func_logits = self.function_selector(x)
        return call_logits, func_logits

class MultimodalModel(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 vision_config, 
                 text_config, 
                 num_functions=100,
                 use_pretrained=False):
        super().__init__()
        
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            print(f"Loading pretrained vision backbone: {vision_config.get('pretrained_model_name')}")
            self.vision_encoder = AutoModel.from_pretrained(vision_config.get('pretrained_model_name'))
            
            if hasattr(self.vision_encoder.config, 'hidden_size'):
                vision_dim = self.vision_encoder.config.hidden_size
            else:
                vision_dim = 1280 
            
            print(f"Loading pretrained text backbone: {text_config.get('pretrained_model_name')}")
            self.text_decoder = AutoModelForCausalLM.from_pretrained(text_config.get('pretrained_model_name'))
            
            if hasattr(self.text_decoder.config, 'n_embd'):
                text_dim = self.text_decoder.config.n_embd
            else:
                text_dim = self.text_decoder.config.hidden_size
        else:
            self.vision_encoder = VisionTransformer(**vision_config)
            vision_dim = vision_config['embed_dim']
            self.text_decoder = TextDecoder(vocab_size, **text_config)
            text_dim = text_config['embed_dim']
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        
        self.projector = MultimodalProjector(vision_dim=vision_dim, text_dim=text_dim)
        self.function_head = FunctionCallingHead(embed_dim=text_dim, num_functions=num_functions)

    def forward(self, input_ids, images=None, targets=None, function_targets=None):
        # 1. Process Images
        if images is not None:
            vision_outputs = self.vision_encoder(images)
            if hasattr(vision_outputs, 'last_hidden_state'):
                vision_features = vision_outputs.last_hidden_state
            elif isinstance(vision_outputs, (list, tuple)):
                vision_features = vision_outputs[0]
            else:
                vision_features = vision_outputs
                
            # Ensure vision_features is [B, Seq, Dim]
            if len(vision_features.shape) == 2: # [B, Dim] -> [B, 1, Dim]
                vision_features = vision_features.unsqueeze(1)
            elif len(vision_features.shape) == 4: # [B, C, H, W] -> [B, H*W, C]
                vision_features = vision_features.flatten(2).transpose(1, 2)
                
            vision_embeddings = self.projector(vision_features)
        else:
            vision_embeddings = None


        # 2. Process Text and Fusion
        if self.use_pretrained:
            # Handle pretrained transformers fusion
            inputs_embeds = self.text_decoder.get_input_embeddings()(input_ids)
            if vision_embeddings is not None:
                full_embeds = torch.cat((vision_embeddings, inputs_embeds), dim=1)
                outputs = self.text_decoder(inputs_embeds=full_embeds, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                text_hidden = hidden_states[:, vision_embeddings.size(1):, :]
                logits = self.text_decoder.lm_head(text_hidden)
            else:
                outputs = self.text_decoder(input_ids=input_ids, output_hidden_states=True)
                text_hidden = outputs.hidden_states[-1]
                logits = outputs.logits
        else:
            # Modern Scratch implementation with RoPE
            tok_emb = self.text_decoder.tok_embeddings(input_ids)
            
            if vision_embeddings is not None:
                x = torch.cat((vision_embeddings, tok_emb), dim=1)
            else:
                x = tok_emb
            
            bsz, seqlen, _ = x.shape
            freqs_cis = self.text_decoder.freqs_cis[:seqlen].to(x.device)
            
            h = x
            for layer in self.text_decoder.layers:
                h = layer(h, freqs_cis)
            
            h = self.text_decoder.norm(h)
            
            if vision_embeddings is not None:
                text_hidden = h[:, vision_embeddings.size(1):, :]
            else:
                text_hidden = h
                
            logits = self.text_decoder.output(text_hidden)

        # 3. Function Head
        last_token_hidden = text_hidden[:, -1, :]
        call_logits, func_logits = self.function_head(last_token_hidden)
        
        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            if function_targets is not None:
                call_loss = loss_fct(call_logits, function_targets['call'])
                func_loss = loss_fct(func_logits, function_targets['func'])
                loss += call_loss + func_loss
                
        return {
            "logits": logits,
            "call_logits": call_logits,
            "func_logits": func_logits,
            "loss": loss
        }

    @torch.no_grad()
    def generate(self, input_ids, images=None, max_new_tokens=100, temperature=1.0, top_p=0.9, top_k=50):
        # Move inputs to device
        device = input_ids.device
        
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids, images=images)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Stop if EOS or Function Call trigger
            if next_token.item() == 2: # Assuming 2 is EOS
                break
                
            call_probs = F.softmax(outputs['call_logits'], dim=-1)
            if torch.argmax(call_probs, dim=-1) == 1:
                break
                
        return input_ids

