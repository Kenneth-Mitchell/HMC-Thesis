import neonwranglerpy
from neonwranglerpy import retrieve_vst_data

savepath = '/content'
vst = retrieve_vst_data('DP1.10098.001', 'DELA', "2018-01", "2022-01", savepath=savepath, save_files=True, stacked_df=True)

vst_data = vst['vst']
columns_to_drop_na = ['plotID', 'siteID', 'utmZone', 'easting', 'northing']
vst_data = vst_data.dropna(subset=columns_to_drop_na)

year = "2021"
site = "GRSM"

dir = "C:\Users\kmitchell\Documents\GitHub\DeepTreeAttention\data\user_data\NEON.D07.GRSM.DP3.30010.001.2021-06.basic.20231214T013758Z.RELEASE-2023"

site_level_data = vst_data[vst_data.plotID.str.contains(site)]
get_tiles = ((site_level_data.easting/1000).astype(int) * 1000).astype(str) + "_" + ((site_level_data.northing/1000).astype(int) * 1000).astype(str)

pattern = fr"{year}_{site}_.*_{get_tiles.unique()[0]}"

for file in files:
    if re.match(pattern, file):
       image_file = os.path.join(root, file)
       print(image_file)