import os
from astroquery.mast import Observations
import pandas as pd
import subprocess

def main():

    subprocess.call("cd ..", shell = True)

    df = pd.read_csv('data_files/valid_points_list.csv')
    df_string = df.astype(str)
    df = df.assign(resolved_coord = df_string['RA'] + " " + df_string['Dec'])

    def save_images_supernova(resolved_coord, i):
        """
        If there are images for the supernovae in JWST or HST at the resolved query, create a folder and keep the data there
        Parameters:
            resolved_coord_index: integer, the index number of the query
        Returns:
            Folder of Data at requested file path (pantheon_data_folder/{resolved coord})
        """
        print(resolved_coord)

        #try the the query, if a error is thrown then print "No Data Points" and "skip" this data point, else make folder and store data
        try:
            obs_table = Observations.query_criteria(coordinates=resolved_coord,
                                            radius="0.006 deg",
                                            intentType = 'science',
                                            filters = ['F1*'],
                                            obs_collection=['HST'])
            obs_table = obs_table[obs_table['calib_level']==3]
            data_products = Observations.get_product_list(obs_table)
            data_products = data_products[data_products['calib_level'] == 3]
            data_products = data_products[data_products['productType'] == 'SCIENCE']
        except: 
            print("No Data Points :(")
            exit
        else:
            print("Has Data Points!")
            folder_path = "pantheon_data_folder/{}".format(df['SNID'][i])
            os.makedirs(folder_path, exist_ok=True)
            manifest = Observations.download_products(data_products, download_dir=folder_path, extension=['fits'])
            print(manifest)
    
    for i in range(len(df['resolved_coord'])):
        save_images_supernova(df['resolved_coord'][i], i)

if __name__=="__main__":
    main()