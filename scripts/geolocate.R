options(stringsAsFactors = F, warn = -1)
library(sp)
library(raster)
library(neonUtilities)
library(geoNEON)
library(yaml)
library(dplyr)
library(tidyr)


# Read the config.yml file
config <- yaml::read_yaml("../config.yml")
sites_of_interest <- config$sites

token <- 'NO TOKEN'
if (file.exists("../API_token")) {
    token <- readLines("../API_token")
}

# Load in situ phenology data
print("Loading in situ phenology data")
vst <- loadByProduct(dpID="DP1.10055.001" ,check.size=F, token=token, site=sites_of_interest)
phe_perindividual <- vst$phe_perindividual

print("Loading in situ location data")
vst <- loadByProduct(dpID="DP1.10098.001", check.size=F, token=token, site=sites_of_interest)
vst_mappingandtagging <- vst$vst_mappingandtagging

# Filter to include sites of interest
print("Filtering to include sites of interest")
phe_perindividual <- dplyr::filter(phe_perindividual, siteID %in% sites_of_interest)
vst_mappingandtagging <- dplyr::filter(vst_mappingandtagging, siteID %in% sites_of_interest)


# Get the exact locations and write to csv 
print("Getting exact locations and writing to csv")
up_phe_perindividual <- geoNEON::getLocTOS(phe_perindividual, 'phe_perindividual', token=token)
up_loc_perindividual <- geoNEON::getLocTOS(vst_mappingandtagging, 'vst_mappingandtagging', token=token)
up_loc_perindividual <- mutate(up_loc_perindividual, adjElevation = as.numeric('adjElevation'))
merged_loc_and_phe <- full_join(up_phe_perindividual, up_loc_perindividual) %>% 
  drop_na(adjDecimalLatitude)

  
write.csv(merged_loc_and_phe, '../temp_data/geo.csv')


# Print the path where the geo data is saved
print(paste("Geo data saved to: /temp_data/geo.csv"))
