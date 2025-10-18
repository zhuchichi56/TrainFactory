from dataclasses import field,dataclass

@dataclass
class DebertaConfig:
    reward_name: str = "/home/admin/data/huggingface_model/other/reward-model-deberta-v3-large-v2"
    max_length: int = 2048
    gpu_num: int = 2
    

@dataclass
class DeitaConfig:
    quality_model_path: str = "/home/admin/data/huggingface_model/other/deita/deita-quality-scorer"
    complexity_model_path: str = "/home/admin/data/huggingface_model/other/deita/deita-complexity-scorer"
    id2score: dict = field(default_factory=lambda: {29896: "1", 29906: "2", 29941: "3", 29946: "4", 29945: "5", 29953: "6"})
    

@dataclass
class InstaggerConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    max_tokens=512
    model_name_or_path: str = "/home/admin/data/huggingface_model/lukeminglkm/instagger_llama2"


@dataclass
class NVConfig:
    gpu_num: int = 8
    
    
    
@dataclass
class TokenConfig:
    model_name: str = "/home/admin/data/huggingface_model/lukeminglkm/instagger_llama2"
    
@dataclass
class pplConfig:
    hf_model_path: str = '/home/admin/data/huggingface_model/LLaMA/llama2-7b'
    gpu_num: int = 2
    
