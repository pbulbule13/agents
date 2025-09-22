import argparse
from euriai import EuriaiClient

def show_model_help():
    print("\n📚 Available Models & Recommendations:\n")
    print(f"{'Provider':<10} {'Model Name':<30} {'ID':<40} {'Best For'}")
    print("-" * 110)
    models = [
        ("OpenAI", "GPT 4.1 Nano", "gpt-4.1-nano", "Fast replies, chatbots"),
        ("OpenAI", "GPT 4.1 Mini", "gpt-4.1-mini", "Smarter gen, code"),
        ("Google", "Gemini 2.5 Pro Exp", "gemini-2.5-pro-exp-03-25", "Complex tasks, code, LLM agents"),
        ("Google", "Gemini 2.0 Flash", "gemini-2.0-flash-001", "Summarization, short Q&A"),
        ("Meta", "Llama 4 Scout", "llama-4-scout-17b-16e-instruct", "Light assistant, ideas"),
        ("Meta", "Llama 4 Maverick", "llama-4-maverick-17b-128e-instruct", "Heavy reasoning, long answers"),
        ("Meta", "Llama 3.3 70B", "llama-3.3-70b-versatile", "Balanced all-round use"),
        ("DeepSeek", "Deepseek R1 Distilled 70B", "deepseek-r1-distill-llama-70b", "Creative, brainstorming"),
        ("Qwen", "Qwen QwQ 32B", "qwen-qwq-32b", "Multilingual, logic"),
        ("Mistral", "Mistral Saba 24B", "mistral-saba-24b", "Summarization, code"),
    ]
    for provider, name, model_id, task in models:
        print(f"{provider:<10} {name:<30} {model_id:<40} {task}")
    
    print("\n🌡️ Suggested Temperatures:")
    print("- 0.2 – 0.4: Deterministic (facts, code)")
    print("- 0.5 – 0.7: Balanced (Q&A, general content) [Default: 0.7]")
    print("- 0.8 – 1.0: Creative (poems, storytelling)")

    print("\n🔢 Suggested Max Tokens:")
    print("- 100–300: Short answers / classification")
    print("- 300–600: Summarization / Q&A")
    print("- 800–2000: Long-form content")

    print("\n💡 Use:")
    print("euriai --api_key <KEY> --prompt 'Hello AI' --model gpt-4.1-nano --temperature 0.7\n")

def main():
    parser = argparse.ArgumentParser(description="Run euriai client")
    parser.add_argument("--api_key", help="Your EURI API Key")
    parser.add_argument("--prompt", help="Prompt to send to the model")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model ID to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=500, help="Max number of tokens")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--models", action="store_true", help="Show all model IDs and suggestions")

    args = parser.parse_args()

    if args.models:
        show_model_help()
        return

    if not args.api_key or not args.prompt:
        parser.error("--api_key and --prompt are required unless using --models")

    client = EuriaiClient(api_key=args.api_key, model=args.model)

    if args.stream:
        print("🔁 Streaming response:\n")
        for chunk in client.stream_completion(args.prompt, temperature=args.temperature, max_tokens=args.max_tokens):
            print(chunk, end='', flush=True)
    else:
        result = client.generate_completion(args.prompt, temperature=args.temperature, max_tokens=args.max_tokens)
        print("✅ Response:\n", result)

if __name__ == "__main__":
    main()
