# import os
# import subprocess


# gpu_ids = [0,1, 2, 3, 4, 5, 6, 7]
# def start_servers(model_path: str, num_processes: int, base_port: int, gpu_ids: list):
#     processes = []
#     gpu_count = len(gpu_ids)

#     for i in range(num_processes):
#         port = base_port + i
#         gpu_id = gpu_ids[i % gpu_count]  # Round-robin assignment of GPUs
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#         print(f"Starting server on port {port} using GPU {gpu_id} and model {model_path}")
#         process = subprocess.Popen(
#             ["python", "vllm_server.py", str(port), model_path],
#             env=os.environ.copy()
#         )
#         processes.append(process)

#     return processes

# def stop_servers(processes):
#     for process in processes:
#         process.terminate()
#         process.wait()
#         print(f"Terminated process with PID {process.pid}")

# if __name__ == "__main__":
#     model_path = "/home/admin/data/huggingface_model/Mistral-7B-Instruct-v0.2"
#     base_port = 8000
#     num_processes = len(gpu_ids)

#     try:
#         processes = start_servers(model_path, num_processes, base_port, gpu_ids)
#         input("Press Enter to stop the servers...\n")  # Keep the servers running
#     finally:
#         stop_servers(processes)
        
        
        
        
import os
import subprocess
import argparse

def start_servers(model_path: str, num_processes: int, base_port: int, gpu_ids: list):
    processes = []
    gpu_count = len(gpu_ids)

    for i in range(num_processes):
        port = base_port + i
        gpu_id = gpu_ids[i % gpu_count]  # Round-robin assignment of GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"Starting server on port {port} using GPU {gpu_id} and model {model_path}")
        process = subprocess.Popen(
            ["python", "vllm_server.py", str(port), model_path],
            env=os.environ.copy()
        )
        processes.append(process)

    return processes

def stop_servers(processes):
    for process in processes:
        process.terminate()
        process.wait()
        print(f"Terminated process with PID {process.pid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start VLLM servers with specified GPUs.")
    parser.add_argument("--model_path", type=str, help="Path to the model.", default="/home/admin/data/huggingface_model/mistral/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--base_port", type=int, help="Starting port number.", default=8000)
    parser.add_argument("--gpu_list", type=str, help="Comma-separated list of GPU IDs.", default="1,2,3,4,5,6,7")
    
    args = parser.parse_args()

    model_path = args.model_path
    base_port = args.base_port
    gpu_ids = [int(gpu) for gpu in args.gpu_list.split(",")]
    num_processes = len(gpu_ids)
    try:
        processes = start_servers(model_path, num_processes, base_port, gpu_ids)
        input("Press Enter to stop the servers...\n")  # Keep the servers running
    finally:
        stop_servers(processes)
        
        