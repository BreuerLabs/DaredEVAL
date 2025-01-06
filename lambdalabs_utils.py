import subprocess
import json
import socket

def execute_shell_command(command):
    try:
        # Execute the shell command
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error while executing command: {e}")
        return None

def process_json_output(output):
    try:
        # Parse the output as JSON
        json_data = json.loads(output)
        return json_data
    except json.JSONDecodeError:
        print("The output is not valid JSON.")
        return None
    
def end_lambdalabs_instance():
    # get current IP address
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    command = "curl -u secret_breuer-labs_c17dc1abad344b1eb25a1388a5d27073.MU4HQ3VE0DND231nvWRoHt3b6SVmC7kZ: https://cloud.lambdalabs.com/api/v1/instances"
    output = execute_shell_command(command)
    all_vms_json = process_json_output(output)

    found_ip = False
    for vm in all_vms_json['data']:
        if vm['ip'] == ip:
            vm_id = vm['id']
            found_ip = True
            break
    
    if not found_ip:
        print("LambdaLabs IP not found, no instance will be terminated")
        return

    command_2 = f"curl -u secret_breuer-labs_c17dc1abad344b1eb25a1388a5d27073.MU4HQ3VE0DND231nvWRoHt3b6SVmC7kZ: https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -d '{{\"instance_ids\":[\"{vm_id}\"]}}'"
    output = execute_shell_command(command_2)
    print(output)

if __name__ == "__main__":
    end_lambdalabs_instance()
    