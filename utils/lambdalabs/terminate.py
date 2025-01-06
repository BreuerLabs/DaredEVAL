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
    
def terminate_lambdalabs_instance():

    # get LL api key
    f = open("utils/lambdalabs/lambdalabs_api_key.txt")
    api_key = f.read()

    # get current IP address
    ip = socket.gethostname().replace("-", ".")

    # search for IP address in current running Lambda Labs instances
    command = f"curl -u {api_key}: https://cloud.lambdalabs.com/api/v1/instances"
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

    # terminate VM if IP address found
    command_2 = f"curl -u {api_key}: https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -d '{{\"instance_ids\":[\"{vm_id}\"]}}'"
    output = execute_shell_command(command_2)
    print(output)

if __name__ == "__main__":
    end_lambdalabs_instance()
    