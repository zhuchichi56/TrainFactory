
# # convert2half.py
# import torch
# import transformers

# def load_model_and_tokenizer(output_dir):
#     model = transformers.AutoModelForCausalLM.from_pretrained(output_dir,   torch_dtype="auto",
#     device_map="auto")
#     tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
#     return model, tokenizer

# def convert_to_fp16_and_save(model, trainer, output_dir,tokenizer):
#     model.half() 
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)


# if __name__ == "__main__":
#     model, tokenizer = load_model_and_tokenizer("/share/home/tj24147/data/trained_model/fanno/llama2/Llama-2-7b-Alpaca52k-GPT4")
#     trainer = transformers.Trainer(model=model)
#     convert_to_fp16_and_save(model, trainer, "/share/home/tj24147/data/trained_model/fanno/llama2/Llama-2-7b-Alpaca52k-GPT4-fp16", tokenizer)
    
 
 # change_hf_format.py
import os
import shutil

def resolve_and_copy_symlinks(model_pth: str, output_pth: str =None):
    if not os.path.exists(model_pth):
        return 

    blobs_pth = os.path.join(model_pth, 'blobs')
    snapshots_pth = os.path.join(model_pth, 'snapshots')
    exact_snapshots_pth = os.path.join(snapshots_pth, os.listdir(snapshots_pth)[0])
    # dst_file_pth = os.path.join(model_pth, item)
    move_dict = []
    for item in os.listdir(exact_snapshots_pth):
        symlink_pth = os.path.join(exact_snapshots_pth, item)
        print(symlink_pth)
        if os.path.islink(symlink_pth):
            target_pth = os.readlink(symlink_pth)
            target_pth = target_pth.split('/')[-1]
            blob_pth = os.path.join(blobs_pth, target_pth)
            symlink_name =  symlink_pth.split('/')[-1]
            move_dict.append({'symlink_file': symlink_name, 'blob_pth': blob_pth})
    

    # copy the blobs to the output
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    
    # change the name of the blob file
    for item in move_dict:
        shutil.copy2(item['blob_pth'], os.path.join(output_pth, item['symlink_file']))
    


# (llf) root@cgh-cpu-0:/volume/pt-train/users/wzhang/ghchen/models# l
# Llama-2-13b/  Llama-2-13b-chat/  Llama-2-7b/  Llama-2-7b-chat/  beaver-7b-v1.0-cost/

# /volume/pt-train/users/wzhang/ghchen/zh/models


# print('Running the function')
# # for model_name in [
# for model_name in os.listdir(''):
resolve_and_copy_symlinks(f'/volume/pt-train/users/wzhang/ghchen/zh/models/models--meta-llama--Llama-2-7b-hf', f'/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b')
# resolve_and_copy_symlinks('/home/admin/.cache/huggingface/hub/models--codellama--CodeLlama-7b-Python-hf','/home/admin/test/CodeLlama-7b-Python-hf')



   