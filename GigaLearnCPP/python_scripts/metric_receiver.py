import site
import sys
import json
import os

wandb_run = None

# Takes in the python executable path, the three wandb init strings, and optionally the current run ID
# Returns the ID of the run (either newly created or resumed)
def init(py_exec_path, project, group, name, id = None):

	global wandb_run
	
	# Fix the path of our interpreter so wandb doesn't run RLGym_PPO instead of Python
	# Very strange fix for a very strange problem
	sys.executable = py_exec_path
	
	try:
		site_packages_dir = os.path.join(os.path.join(os.path.dirname(py_exec_path), "Lib"), "site-packages")
		sys.path.append(site_packages_dir)
		site.addsitedir(site_packages_dir)
		import wandb
	except Exception as e:
		print(f"[metric_receiver] ERROR: failed to import wandb. Python: {py_exec_path}")
		print(f"[metric_receiver] site-packages: {site.getsitepackages()}")
		print(f"[metric_receiver] Exception: {repr(e)}")
		raise Exception(f"""
			FAILED to import wandb! Make sure GigaLearnCPP isn't using the wrong Python installation.
			This installation's site packages: {site.getsitepackages()}
			Exception: {repr(e)}"""
		)
	
	print("Calling wandb.init()...")
	try:
		if not (id is None) and len(id) > 0:
			wandb_run = wandb.init(project = project, group = group, name = name, id = id, resume = "allow")
		else:
			wandb_run = wandb.init(project = project, group = group, name = name)
	except Exception as e:
		print("[metric_receiver] ERROR: wandb.init failed.")
		print(f"[metric_receiver] project={project} group={group} name={name} id={id}")
		print(f"[metric_receiver] Exception: {repr(e)}")
		raise
	return wandb_run.id

def add_metrics(metrics):
	global wandb_run
	if wandb_run is None:
		print("[metric_receiver] ERROR: wandb_run is None; did init() fail?")
		return
	try:
		wandb_run.log(metrics)
	except Exception as e:
		print("[metric_receiver] ERROR: wandb_run.log failed.")
		print(f"[metric_receiver] metrics keys: {list(metrics.keys())[:10]}")
		print(f"[metric_receiver] Exception: {repr(e)}")
		raise
