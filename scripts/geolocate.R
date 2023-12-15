options(stringsAsFactors = F, warn = -1)
library(sp)
library(raster)
library(neonUtilities)
library(geoNEON)
library(yaml)
library(dplyr)


# Read the config.yml file
config <- yaml::read_yaml("config.yml")
sites_of_interest <- config$sites

token <- 'NO TOKEN'
if (file.exists("API_token.txt")) {
    token <- readLines("API_token.txt")
}

# Load in situ phenology data
print("Loading in situ phenology data")
vst <- loadByProduct(dpID="DP1.10055.001" ,check.size=F, token=token, site=sites_of_interest)
phe_perindividual <- vst$phe_perindividual

# Filter to include sites of interest
print("Filtering to include sites of interest")
phe_perindividual <- dplyr::filter(phe_perindividual, siteID %in% sites_of_interest)

# Get the exact locations and write to csv 
print("Getting exact locations and writing to csv")
up_phe_perindividual <- geoNEON::getLocTOS(phe_perindividual, 'phe_perindividual', token=token)
write.csv(up_phe_perindividual, 'temp_data/geo.csv')

# Print the path where the geo data is saved
print(paste("Geo data saved to:", getwd(), "/temp_data/geo.csv"))
