def freeze_model_for_stage(model, stage):
    """
    Utility to freeze/unfreeze model parts based on the training stage.
    """
    if stage == "pretrain_vision":
        # Only train vision encoder
        for param in model.parameters():
            param.requires_grad = False
        for param in model.vision_encoder.parameters():
            param.requires_grad = True
            
    elif stage == "pretrain_text":
        # Only train text decoder
        for param in model.parameters():
            param.requires_grad = False
        for param in model.text_decoder.parameters():
            param.requires_grad = True
            
    elif stage == "alignment":
        # Freeze encoders, train only the projector
        for param in model.parameters():
            param.requires_grad = False
        for param in model.projector.parameters():
            param.requires_grad = True
            
    elif stage == "instruction_tuning":
        # Train everything (or use LoRA)
        for param in model.parameters():
            param.requires_grad = True
            
    elif stage == "function_calling":
        # Focus on function head and decoder
        for param in model.parameters():
            param.requires_grad = False
        for param in model.function_head.parameters():
            param.requires_grad = True
        for param in model.text_decoder.parameters():
            param.requires_grad = True

    return model
