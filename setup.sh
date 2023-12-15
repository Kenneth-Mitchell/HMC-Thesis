#!/bin/bash
# Create input_data and temp_data directories
echo "Creating directories..."
mkdir -p input_data temp_data


if [[ ! -f "API_token" ]]; then
    read -p "Do you have an NEON API token? (y/n): " use_token
    if [[ $use_token == "y" ]]; then
        read -p "Enter your NEON API token: " api_token
        echo "$api_token" > API_token
    else
        echo 
        echo "It is recommended to obtain an API token to speed up the geolocation process."
        echo "If you obtain one in the future, just run setup.sh again."
        echo "To obtain an NEON API token, please visit: https://data.neonscience.org/data-api/"
        echo 
    fi
fi

# Function to install an R package using remotes
install_r_package() {
    package_name=$1
    echo "Installing R package: $package_name"
    Rscript -e "if (!requireNamespace('remotes', quietly = TRUE)) install.packages('remotes', repos = 'http://cran.us.r-project.org'); remotes::install_github('$package_name')"
}

# Function to install a Python package from GitHub
install_python_package() {
    package_name=$1
    echo "Installing Python package: $package_name"
    pip install $package_name
}

# Install geoNEON package
echo "Installing geoNEON package..."
install_r_package "NEONScience/NEON-geolocation/geoNEON"

# Install neonwranglerpy package
echo "Installing neonwranglerpy package..."
install_python_package "git+https://github.com/weecology/neonwranglerpy.git"

echo "Setup completed successfully."
