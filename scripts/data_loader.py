
import pandas as pd
import yaml
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# Initialize R packages
neonUtilities = importr('neonUtilities')
base = importr('base')

# Load configuration and token
with open('../config.yml', 'r') as file:
    config = yaml.safe_load(file)

with open('../API_token', 'r') as file:
    token = file.read()




def prepare_phenogeo():
    phenogeo_csv = "/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/phenogeo.csv"
    df = pd.read_csv(phenogeo_csv)
    df = pd.read_csv(phenogeo_csv).dropna(subset=['adjNorthing', 'adjEasting', 'uid'])
    df = df.drop_duplicates(subset='uid')
    df.reset_index(drop=True, inplace=True)
    return df

def unique_locations(df):
    #for each site, return the unique locations (within 1000 utm units)
    df_site = df[['siteID','adjNorthing', 'adjEasting']].copy()
    df_site.loc[:, ['adjNorthing', 'adjEasting']] = df_site[['adjNorthing', 'adjEasting']].round(decimals=-3)
    df_site = df_site.drop_duplicates()
    
    
    return df_site

# @dask.delayed
def download_data_chunk(dpID, site, year, Easting, Northing, buffer=0):
    #given the dpID and site, download the data (either rgb or hsi)
    neonUtilities.byTileAOP(dpID, site, year, Easting, Northing, buffer, True, False)
    pass

def main():
    df = prepare_phenogeo()
    df_site = unique_locations(df)

    rgb_dpID = config['dpIDs'][0]
    hsi_dpID = config['dpIDs'][1]

    years_by_site = {}
    for site in df_site['siteID']:
        years_by_site[site] = []

    for batch in config['batches']:
        year = list(batch.items())[0][0]
        sites = list(batch.items())[0][1]
        for site in sites:
            years_by_site[site].append(year)

    for row in df_site.itertuples():
        site = row.siteID
        Easting = row.adjEasting
        Northing = row.adjNorthing
        for year in years_by_site[site]:
            download_data_chunk(rgb_dpID, site, year, Easting, Northing)
            download_data_chunk(hsi_dpID, site, year, Easting, Northing)


if __name__ == '__main__':
    main()