import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(model_path: str, question: str):
    """
    Visualize attention scores for LLaMA model on GSM8K questions.
    
    Args:
        model_path: Path to LLaMA model
        question: GSM8K question text
    """
    # Load model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True  # Enable attention outputs
    )

    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    # Generate with attention outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_return_sequences=1,
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    # Get generated tokens and attention scores
    generated_ids = outputs.sequences[0] # shape: (seq_len,)
    print(outputs.attentions[0].shape) # shape: (layers, heads, seq_len, seq_len)
    attention_scores = torch.stack(outputs.attentions).squeeze() # shape: (layers, heads, seq_len, seq_len)
    

    #  outputs.attentions  # tuple of tensors, 每个tensor形状: (layers, heads, seq_len, seq_len)
    # ↓ torch.stack()     # 将tuple中的tensors堆叠
    # ↓ squeeze()         # 移除大小为1的维度
    # attention_scores    # 最终形状: (layers, heads, seq_len, seq_len)
    
    # Average attention scores across heads and layers
    avg_attention = attention_scores.mean(dim=(0,1)).cpu().numpy()
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    
    # Create attention heatmap
    plt.figure(figsize=(15,10))
    sns.heatmap(
        avg_attention[:len(tokens), :len(tokens)],
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd'
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Average Attention Scores Across Heads and Layers")
    
    # Find tokens with highest attention scores
    top_k = 5
    max_attention_scores = avg_attention.max(axis=1)
    top_token_indices = np.argsort(max_attention_scores)[-top_k:]
    
    print("\nTokens with highest attention scores:")
    for idx in top_token_indices:
        print(f"Token: {tokens[idx]}, Score: {max_attention_scores[idx]:.4f}")
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    question = """
    Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning 
    and sells the rest at the farmers market daily for $2 per egg. 
    How much in dollars does she make per week?
    """
    
    visualize_attention("meta-llama/Llama-2-7b-chat-hf", question)
