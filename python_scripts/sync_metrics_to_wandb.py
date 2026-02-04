import struct
import time
import os
import sys
import wandb

def read_string(f):
    length = struct.unpack('I', f.read(4))[0]
    return f.read(length).decode('utf-8')

def sync_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Waiting for metrics file {file_path}...")
        while not os.path.exists(file_path):
            time.sleep(1)
    
    with open(file_path, 'rb') as f:
        # Read Header
        magic = f.read(4).decode('utf-8')
        if magic != 'GGLM':
            print("Invalid metrics file format.")
            return

        version = struct.unpack('I', f.read(4))[0]
        project = read_string(f)
        group = read_string(f)
        name = read_string(f)
        run_id = read_string(f)

        print(f"Syncing Run: {project}/{group}/{name} (ID: {run_id})")
        
        wandb.init(
            project=project,
            group=group,
            name=name,
            id=run_id,
            resume="allow"
        )

        while True:
            pos = f.tell()
            data = f.read(8)
            if not data:
                time.sleep(0.5)
                f.seek(pos)
                continue
            
            timestamp = struct.unpack('d', data)[0]
            num_metrics = struct.unpack('I', f.read(4))[0]
            
            metrics = {}
            for _ in range(num_metrics):
                key = read_string(f)
                val = struct.unpack('f', f.read(4))[0]
                metrics[key] = val
            
            wandb.log(metrics)
            print(f"Logged {num_metrics} metrics at {timestamp}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sync_metrics_to_wandb.py <path_to_gglm_file>")
        sys.exit(1)
    
    try:
        sync_metrics(sys.argv[1])
    except KeyboardInterrupt:
        print("Sync stopped.")
    except Exception as e:
        print(f"Error: {e}")
