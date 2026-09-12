#!/usr/bin/env python3
"""
Quick verification script to test gradient flow with gradient checkpointing + LoRA.
This verifies the fix for the "element 0 of tensors does not require grad" error.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def test_gradient_flow():
    """Test that gradients flow correctly through LoRA params with gradient checkpointing."""
    print("=" * 60)
    print("Testing Gradient Flow with Gradient Checkpointing + LoRA")
    print("=" * 60)

    # Use a small model for quick testing
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n1. Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # Apply LoRA
    print("\n2. Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    print("\n3. Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # Prepare test input
    print("\n4. Preparing test input...")
    text = "What is 2 + 2? The answer is 4."
    encodings = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

    # Test the approach from the fix: temporarily disable gradient checkpointing
    print("\n5. Testing forward pass with temporary gradient checkpointing disable...")

    was_checkpointing = getattr(model, 'gradient_checkpointing', False)
    print(f"   Gradient checkpointing was enabled: {was_checkpointing}")

    if was_checkpointing and hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("   Temporarily disabled gradient checkpointing")

    try:
        outputs = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
        )
        print("   Forward pass successful!")
    finally:
        if was_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("   Re-enabled gradient checkpointing")

    # Compute loss
    print("\n6. Computing loss...")
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = encodings["input_ids"][:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    # Simple loss: negative mean log prob (like in policy gradient)
    loss = -token_log_probs.mean()
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Loss requires_grad: {loss.requires_grad}")

    # Test backward pass
    print("\n7. Testing backward pass...")
    try:
        loss.backward()
        print("   Backward pass successful!")
    except RuntimeError as e:
        print(f"   ERROR: {e}")
        return False

    # Check that LoRA parameters received gradients
    print("\n8. Checking LoRA parameter gradients...")
    lora_params_with_grad = 0
    lora_params_without_grad = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    lora_params_with_grad += 1
                    if lora_params_with_grad <= 3:  # Show first 3
                        print(f"   ✓ {name}: grad_norm = {grad_norm:.6f}")
            else:
                lora_params_without_grad += 1

    print(f"\n   LoRA params with gradients: {lora_params_with_grad}")
    print(f"   LoRA params without gradients: {lora_params_without_grad}")

    if lora_params_with_grad > 0 and lora_params_without_grad == 0:
        print("\n" + "=" * 60)
        print("✓ SUCCESS: All LoRA parameters received gradients!")
        print("  The gradient fix is working correctly.")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("✗ FAILURE: Not all LoRA parameters received gradients.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    try:
        success = test_gradient_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
