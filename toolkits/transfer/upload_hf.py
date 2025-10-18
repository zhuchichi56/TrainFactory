from transformers import AutoModelForCausalLM,AutoTokenizer

access_token = "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"

# model = AutoModelForCausalLM.from_pretrained("/home/admin/data/ppl_output1/llama-2-7b-0322_with_direct_response_v1-ft/merge/M0")
tokenizer = AutoTokenizer.from_pretrained("/home/admin/data/ppl_output1/llama-2-7b-0322_with_direct_response_v1-ft/merge/M0")

# model.push_to_hub("ShadowFall09/tyc_test1",token =access_token)
tokenizer.push_to_hub("ShadowFall09/tyc_test1",token =access_token)

