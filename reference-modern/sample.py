"""
Generate text from a trained model.

Usage:
    python sample.py --prompt "ROMEO:" --max_tokens 200 --temperature 0.8
"""

import argparse
import torch

from config import Config
from model import DecoderOnlyTransformer
from data import CharTokenizer


def load_model_and_tokenizer(ckpt_path: str, device: str = "cpu") -> tuple:
    """Load a trained model and its tokenizer from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer: CharTokenizer = checkpoint["tokenizer"]

    # Make sure vocab size matches
    config.model.vocab_size = tokenizer.vocab_size

    model = DecoderOnlyTransformer(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="ROMEO:",
                        help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0-2.0)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (None to disable)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model, tokenizer, config = load_model_and_tokenizer(args.ckpt, device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Encode prompt
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # Decode and print
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)
    print("-" * 60)
    print(f"\nGenerated {len(generated[0]) - len(prompt_ids[0])} tokens")


if __name__ == "__main__":
    main()
