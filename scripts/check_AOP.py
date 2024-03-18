import os
import requests
import json
import itertools
from itertools import chain
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import math

SERVER = 'http://data.neonscience.org/api/v0/'

# Read the config.yml file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Get the sites from the config
sites = config['sites']

# Set the data path
input_path = 'input_data/phenology'
output_path = 'temp_data'
phenogeo_path = os.path.join(output_path, 'phenogeo.csv')

phenogeo = pd.read_csv(phenogeo_path)

site_months = {}

for site in sites:
    SITECODE = site

    #Define the url, using the sites/ endpoint
    url = SERVER+'sites/'+SITECODE

    #Request the url
    site_request = requests.get(url)

    #Convert the request to Python JSON object
    site_json = site_request.json()

    #Set the Ecosystem structure (CHM) data product
    # PRODUCTCODE = 'DP1.30001.001' # LiDAR slant range waveform
    # PRODUCTCODE = 'DP1.30003.001' # Discrete return LiDAR point cloud
    # PRODUCTCODE = 'DP3.30010.001' # High-resolution orthorectified camera imagery mosiac
    PRODUCTCODE = 'DP3.30010.001' # High-resolution orthorectified camera imagery



    #Get available months of Ecosystem structure data products for TEAK site
    #Loop through the 'dataProducts' list items (each one is a dictionary) at the site
    for product in site_json['data']['dataProducts']: 
        #if a list item's 'dataProductCode' dict element equals the product code string
        if(product['dataProductCode'] == PRODUCTCODE): 
            #print the available months
            print(site)
            print('Available Months: ',product['availableMonths'])
            months = [pd.to_datetime(month).strftime('%Y-%m') for month in product['availableMonths']]
            site_months[site] = months
            print('URLs for each Month:')
            #print the available URLs
            for url in product['availableDataUrls']:
                print(url)

# Load the data
phenogeo = phenogeo[phenogeo['taxonID_x'] == 'LITU']
# Convert 'date_y' to datetime
phenogeo['date_y'] = pd.to_datetime(phenogeo['date_y'])

# Create a new column
phenogeo['Flower_Status'] = np.nan

# Iterate over each site and its AOP flyover months
for site, months in site_months.items():
    for month in months:
        # Get the AOP flyover date
        aop_date = pd.to_datetime(month)

        # Filter the dataframe for the site
        site_df = phenogeo[phenogeo['siteID'] == site]

        # Group the data by 'individualID'
        grouped = site_df.groupby('individualID')

        # Iterate over each group
        for name, group in grouped:
            # Calculate the absolute difference between the observation dates and the AOP date
            group['date_diff'] = (group['date_y'] - aop_date).abs()

            # Identify the observation closest to the AOP date
            closest_observation = group.loc[group['date_diff'].idxmin()]

            # Check if the 'phenophaseName' is 'Open flowers'
            if closest_observation['phenophaseName'] == 'Open flowers':
                phenogeo.at[closest_observation.name, 'Flower_Status'] = 'Flowering'
            else:
                # Filter the group to include only the observations where 'phenophaseName' is 'Open flowers'
                flowering_group = group[group['phenophaseName'] == 'Open flowers']

                # Calculate the absolute difference between the flowering observation dates and the AOP date
                flowering_group.loc[:, 'date_diff'] = (flowering_group['date_y'] - aop_date).abs()

                # Identify the closest flowering observation to the AOP date
                closest_flowering_observation = flowering_group.loc[flowering_group['date_diff'].idxmin()]

                # Check if the closest flowering observation date is before or after the AOP date
                if closest_flowering_observation['date_y'] < aop_date:
                    phenogeo.at[closest_flowering_observation.name, 'Flower_Status'] = 'Already Flowered'
                else:
                    phenogeo.at[closest_flowering_observation.name, 'Flower_Status'] = 'Yet to Flower'
# Convert 'date_y' to year
phenogeo['year'] = phenogeo['date_y'].dt.year

# Group the data by 'siteID', 'year', and 'Flower_Status'
grouped = phenogeo.groupby(['siteID', 'year', 'Flower_Status'])

# Count the number of trees for each group
counts = grouped.size().unstack(fill_value=0)

# Get the unique sites
sites = phenogeo['siteID'].unique()

# Calculate the number of rows and columns for the subplots
rows = math.ceil(len(sites) / 2)
cols = 2 if len(sites) > 1 else 1

# Calculate the maximum count
max_count = counts.max().max() + 20

# Create a figure with multiple subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))

# Flatten the axes
axes = axes.flatten()

# Iterate over each site
for i, (site, data) in enumerate(counts.groupby(level=0)):
    # Reset the index of 'data'
    data.reset_index(level=0, drop=True, inplace=True)

    # Plot the data as a stacked bar plot on the current subplot
    data.plot(kind='bar', stacked=True, ax=axes[i])

    # Set the title and labels
    axes[i].set_title(f'Flowering Status of Tulip Trees at {site} during AOP flyover date')
    axes[i].set_xlabel('Year')
    axes[i].set_ylabel('Count')

    # Set the y-axis limit
    axes[i].set_ylim([0, max_count])

# Remove the extra subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Save the plot as an image file
plt.savefig('temp_data/check_AOP.png')

# Show the plot
plt.tight_layout()
plt.tight_layout()
plt.show()
