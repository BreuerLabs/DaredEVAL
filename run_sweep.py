import os
import subprocess
import argparse
import yaml

def run_wandb_sweep(sweep_yaml_path, project_name, num_agents=1):
    """
    Automates the creation and execution of a Weights & Biases sweep.

    Parameters:
        sweep_yaml_path (str): Path to the sweep YAML configuration file.
        project_name (str): Name of the WandB project.
        num_agents (int): Number of agents to run in parallel.
    """
    # Check if YAML file exists
    if not os.path.exists(sweep_yaml_path):
        print(f"Error: The file {sweep_yaml_path} does not exist.")
        return

    # Read the sweep YAML
    with open(sweep_yaml_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Ensure "program" is in the YAML
    if "program" not in sweep_config:
        print("Error: The sweep YAML file must include a 'program' field.")
        return

    # Run the sweep initialization
    print(f"Initializing WandB sweep for project '{project_name}'...")
    result = subprocess.run(["wandb", "sweep", sweep_yaml_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error initializing the sweep:\n{result.stderr}")
        return

    # Extract the sweep ID from the command output
    output = result.stdout.strip()
    print(output)
    sweep_id = output.split("/")[-1]
    print(f"Sweep initialized with ID: {sweep_id}")

    # Launch the specified number of agents
    for i in range(num_agents):
        print(f"Launching agent {i + 1}/{num_agents}...")
        subprocess.Popen(["wandb", "agent", f"{project_name}/{sweep_id}"])

    print("Sweep agents are running. Check the WandB dashboard for progress.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a WandB sweep with multiple agents.")
    parser.add_argument("sweep_yaml", type=str, help="Path to the sweep YAML configuration file.")
    parser.add_argument("project_name", type=str, help="Name of the WandB project.")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of agents to run in parallel (default: 1).")

    args = parser.parse_args()

    # Run the sweep
    run_wandb_sweep(args.sweep_yaml, args.project_name, args.num_agents)