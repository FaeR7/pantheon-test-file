from astropy.io import ascii
import pandas as pd
import numpy as np
import glob
import os
import pathlib
import subprocess


def create_radius_index(root_dir, relative_key, output_csv):
    """
    Create a datatable with the RADIUS of objects of around the RA and DEC of each SN1 source extractor file
    Parameters:
        root_dir (str): The root directory to search.

    Returns:
        dict: A dictionary where keys are relative paths to .cat files,
              and values are dictionaries of the first data row.
    """
    # Find all .cat files recursively
    
    cat_files = glob.glob(os.path.join(root_dir, '**', '*.cat'), recursive=True)

    data_dict = []

    for file_path in cat_files:
        try:
            starcounts = ascii.read(file_path, data_start=0)

            strPath = os.path.realpath(file_path)
            nmFolders = strPath.split(os.path.sep)

            SNID_key = nmFolders[-3]
            file_key = nmFolders[-2]
            relative_key += 1

            sn1_data = pd.read_csv('data_files/unmatched_pantheon.csv')

            ra_sn1 = sn1_data.loc[sn1_data['SNID'] == SNID_key, 'RA'].values[0]
            dec_sn1 = sn1_data.loc[sn1_data['SNID'] == SNID_key, 'Dec'].values[0]

            for i in range(len(starcounts)):
                ra_source = starcounts['ALPHA_J2000'][i]
                dec_source = starcounts['DELTA_J2000'][i]
                if ((ra_source - ra_sn1)**2 + (dec_source - dec_sn1)**2 < 5): 
                    info = {
                            "SNID": SNID_key,
                            "File_key": file_key, 
                            "Galaxy_Key": relative_key,
                            "RA_Galaxy": ra_source,
                            "Dec_Galaxy": dec_source,
                            "Position_Angle": starcounts['THETA_IMAGE'][i],
                            "Major_Axis": starcounts['A_IMAGE'][i],
                            "Minor_Axis": starcounts['B_IMAGE'][i],
                            "Mag": starcounts['MAG_AUTO'][i],
                            "Mag_Error": starcounts['MAGERR_AUTO'][i],
                            "Flux": starcounts['FLUX_AUTO'][i],
                            "Flux_Error": starcounts['FLUXERR_AUTO'][i]
                            }
                    data_dict.append(info)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        # Convert to DataFrame and export
    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv, index=False)
    print(f"CSV exported: {output_csv}")

    return data_dict





if __name__== "__main__":
    subprocess.call("cd ..", shell = True)
    home_dir = os.getcwd()
    ## Important for me, only look in source_extractor file
    root_dir = os.path.join(home_dir, 'source_extractor')
    print(root_dir)
    relative_key = 0
    create_radius_index(root_dir, relative_key, "radius_galaxy.csv")