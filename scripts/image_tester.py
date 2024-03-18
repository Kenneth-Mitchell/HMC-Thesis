from image import Image
#read in phenogeo csv
import pandas as pd
phenogeo_csv = "/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/temp_data/phenogeo.csv"
df = pd.read_csv(phenogeo_csv)
df = pd.read_csv(phenogeo_csv).dropna(subset=['adjNorthing', 'adjEasting', 'uid'])
df = df.drop_duplicates(subset='uid')
df.reset_index(drop=True, inplace=True)

sample_image_path = "/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/rgb_test/2021_GRSM_5_275000_3951000_image.tif"

img = Image(sample_image_path)
img.annotate(df)
img.get_bounding_boxes()

img.generate_hsi_trees("/mnt/c/Users/kmitchell/Documents/GitHub/Thesis/input_data/hsi/NEON_D07_GRSM_DP3_275000_3951000_reflectance.h5")
print(img.gdf.head())
