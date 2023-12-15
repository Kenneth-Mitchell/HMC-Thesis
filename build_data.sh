#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Define paths to output files and scripts
output_geolocate="temp_data/geo.csv"
geolocate_script="scripts/geolocate.R"
geolocate_rerun=false

input_phenology="input_data/phenology/"
output_phenogeo="temp_data/phenogeo.csv"
phenogeo_script="scripts/phenogeo.py"
phenogeo_rerun=false


# Check for dependencies
if ! command -v Rscript &> /dev/null
then
    echo "Rscript could not be found"
    exit
fi

if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found"
    exit
fi

# Check rerun all
if [ "$1" == "rerun_all" ]; then
    echo "Are you sure you want to rerun all scripts (this will take a long time) (y/n)"
    read answer
    if [ "$answer" == "y" ]; then
        geolocate_rerun=true
        phenogeo_rerun=true
    fi
fi

### GEOLOCATE ###

# Extracts exact latitude and longitude of plants
# for sites specified in config.yml using the NEON API.

# Note: Without a NEON API key, this script takes around 2 hours to run,
# due to the rate limiting. I suggest only running this if absolutely
# needed, and instead using the previously generate output file.

# Check for rerun flag
if [ "$1" == "geolocate_rerun" ]  || [ "$1" == "rerun_all" ]; then
    geolocate_rerun=true
fi

# Run geolocate.R if output does not exist or rerun flag is specified
if [ ! -f "$output_geolocate" ] || [ "$geolocate_rerun" = true ]; then
    if [ ! -f "$geolocate_script" ]; then
        echo "Script $geolocate_script not found"
        exit 1
    fi
    echo "Running $geolocate_script"
    Rscript $geolocate_script
fi
### END GEOLOCATE ###

### PHENOGEO ###
# Merges the phenology and geolocation data.

# Check for rerun flag
if [ "$1" == "phenogeo_rerun" ] || [ "$1" == "rerun_all" ]; then
    phenogeo_rerun=true
fi

# Run phenogeo.py if output does not exist or rerun flag is specified
if [ ! -f "$output_phenogeo" ] || [ "$phenogeo_rerun" = true ]; then

    if [ ! -f "$phenogeo_script" ]; then
        echo "Script $phenogeo_script not found"
        exit 1
    fi

    if [ -z "$(ls -A $input_phenology)" ]; then
        echo "The input_phenology folder is empty"
        exit 1
    fi

    echo "Running $phenogeo_script"
    python3 $phenogeo_script
fi
### END PHENOGEO ###

