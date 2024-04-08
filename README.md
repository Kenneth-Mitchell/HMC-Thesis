# Kenneth Mitchell Thesis HMC '24

## Description
For background information, see the [project proposal](background/kmproposal.pdf).

---
## Set up

### Cloning the Repository

If applicable, clone the repository to your local machine:

```bash
git clone https://github.com/Kenneth-Mitchell/HMC-Thesis
cd HMC-Thesis
```

### Activate the Conda Environment

Use the [`environment.yml`](enivornment.yml) file to create the Conda environment:

```bash
conda env create -f environment.yml
conda activate micropheno
```

### Post-Installation Setup

After activating the `micropheno` environment, run the [`setup.sh`](setup.sh) script to create necessary directories and install more packages:

```bash
setup.sh
```

The [`setup.sh`](setup.sh) script will perform the following actions:

1. Create `input_data` and `temp_data` directories.

2. Prompt user for NEON API token. While optional, I highly suggest using a [API token](https://data.neonscience.org/data-api/).

3. Install the [`geoNEON`](https://github.com/NEONScience/NEON-geolocation/tree/master/geoNEON) package and the [`neonwranglerpy`](https://github.com/weecology/neonwranglerpy) package. Both are unavailable through normal conda channels and thus are not included in [`environment.yml`](enivornment.yml).

### Downloading Data

1. **Download Phenology Data**: 

   Download data product `DP1.10055.001` for the sites and years of interest from the [NEON Data Portal](https://data.neonscience.org/data-products/DP1.10055.001). Place the data under `input_data\phenology`. No need to modify the file structure, for instance, if your downloaded folder is named `DP1.10055.001`, rename it:
   
   ```bash
   mv DP1.10055.001 input_data/phenology
   ```

2. **Update [`config.yml`](config.yml)**: 

   Update [`config.yml`](config.yml) in the project directory to include the downloaded sites, for instance:
   ```yml
   sites:
    - SCBI
    - ORNL
    - GRSM
    - SERC
   ```
---
## Usage

#### [`build_data.sh`](build_data.sh)

Orchestrates the overall phenology data processing:

1. [`geolocate.R`](scripts/geolocate.R) produces the exact geolocation data for the sites in [`config.yml`](config.yml) &rarr; `geo.csv`
2. [`phenogeo.py`](scripts/phenogeo.py) merges the phenology and geolocation data &rarr; `phenogeo.csv`

#### [`check_AOP.py`](scripts/check_AOP.py)

Produces a figure of the flowering status for the sites in [`config.yml`](config.yml) during AOP flyover &rarr; `check_AOP.png`