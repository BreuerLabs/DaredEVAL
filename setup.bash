#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Filesystem setup
echo "Cloning required repositories..."
git clone https://github.com/ShailenSmith/Defend_MI.git
git clone https://github.com/RasmusTorp/Plug_and_Play_Attacks.git
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

# Clean up the stylegan2-ada-pytorch repository
rm -rf stylegan2-ada-pytorch/.git/
rm -rf stylegan2-ada-pytorch/.github/
rm -f stylegan2-ada-pytorch/.gitignore

# Download pretrained file for StyleGAN2
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P stylegan2-ada-pytorch/
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl -P stylegan2-ada-pytorch/

# Add secret data
mkdir -p utils/lambdalabs
echo "secret_breuer-labs_c17dc1abad344b1eb25a1388a5d27073.MU4HQ3VE0DND231nvWRoHt3b6SVmC7kZ" > utils/lambdalabs/lambdalabs_api_key.txt

# Step 2: Rest of setup
echo "Pulling latest updates..."
git pull

# Create a screen session
screen -S pnp -dm bash

# Install necessary packages and set up the environment
echo "Installing required packages..."
sudo apt -y install python3.11-venv python3.11-dev
python3.11 -m venv venv
source ./venv/bin/activate

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

echo "Setup complete!"