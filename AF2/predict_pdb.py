import os
import subprocess
import multiprocessing as mp
from queue import Queue
import threading

def run_colabfold_single(fasta_info):
    """Run a single ColabFold task on the specified GPU"""
    fasta_path, out_path, gpu_id = fasta_info

    print(f"Running ColabFold for {os.path.basename(fasta_path)} on GPU {gpu_id}...")

    # Set the CUDA_VISIBLE_DEVICES environment variable to specify the GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        "colabfold_batch",
        fasta_path,
        out_path,
        "--model-type", "alphafold2_multimer_v3",
        "--num-seeds", "5",
        "--num-recycle", "3",
        "--use-gpu-relax",
        "--num-models", "6",
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"Finished {os.path.basename(fasta_path)} successfully on GPU {gpu_id}.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {os.path.basename(fasta_path)} on GPU {gpu_id}:\n{e}\n")
        return False


def run_colabfold_on_all_fastas_4gpu(fasta_dir, output_dir, num_gpus=4):
    """Run ColabFold in parallel using 4 GPUs"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all FASTA files
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(".fasta") or f.endswith(".fa")]
    print(f"Found {len(fasta_files)} fasta files in '{fasta_dir}'.")

    if not fasta_files:
        print("No FASTA files found!")
        return

    # Prepare task list
    tasks = []
    for i, fasta in enumerate(fasta_files):
        fasta_path = os.path.join(fasta_dir, fasta)
        out_path = os.path.join(output_dir, fasta.rsplit(".", 1)[0])
        gpu_id = i % num_gpus  # Assign GPU in a round-robin manner
        tasks.append((fasta_path, out_path, gpu_id))

    # Use process pool to handle tasks in parallel
    print(f"Starting parallel processing with {num_gpus} GPUs...")
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(run_colabfold_single, tasks)

    # Summarize results
    successful = sum(results)
    failed = len(results) - successful
    print(f"\nProcessing completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


# Alternative version using a task queue
def worker_process(task_queue, gpu_id):
    """Worker process function"""
    while True:
        try:
            task = task_queue.get_nowait()
            if task is None:
                break

            fasta_path, out_path = task
            run_colabfold_single((fasta_path, out_path, gpu_id))
            task_queue.task_done()

        except:
            break


def run_colabfold_queue_version(fasta_dir, output_dir, num_gpus=4):
    """Run ColabFold using a queue-based system"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all FASTA files
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(".fasta") or f.endswith(".fa")]
    print(f"Found {len(fasta_files)} fasta files in '{fasta_dir}'.")

    # Create task queue
    task_queue = mp.Queue()
    for fasta in fasta_files:
        fasta_path = os.path.join(fasta_dir, fasta)
        out_path = os.path.join(output_dir, fasta.rsplit(".", 1)[0])
        task_queue.put((fasta_path, out_path))

    # Create worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(task_queue, gpu_id))
        p.start()
        processes.append(p)

    # Wait for all tasks to finish
    for p in processes:
        p.join()

    print("All tasks completed!")


if __name__ == "__main__":
    # input: directory containing FASTA files with alphafold multimer v3 version
    input_dir = "..."
    output_dir = "..."

    # Run on 4 GPUs
    run_colabfold_on_all_fastas_4gpu(input_dir, output_dir, num_gpus=4)
    # queue-based version
    # run_colabfold_queue_version(input_dir, output_dir, num_gpus=4)
