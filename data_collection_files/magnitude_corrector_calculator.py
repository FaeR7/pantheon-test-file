import pandas as pd 
from astropy.io import fits
import glob
import os
from pathlib import Path
import numpy as np
import subprocess


def main():
    subprocess.call("cd ..", shell = True)
    print(os.getcwd())
    snids = get_snid_folders()
    print(len(snids))
    extract_fits_by_snid(snids, output_csv="corrected_mag.csv", recursive=True)


def extract_fits_by_snid(snid_list, base_path="pantheon_data_folder", output_csv="corrected_mag.csv", recursive=True, data_list = []):
    """
    Loops over SNIDs, reads .fits files in /source_extractor/{SNID}/ folders
        - Extracts the Telescope, Instrument Used, and Filter
    Parameters:
    - snid_list: list of str, SNID folder names
    - base_path: str, base directory containing SNID subfolders (defaults to source extractor)
    - output_csv: str, name of output CSV file (defaults to corrected_mag.csv)
    - recursive: bool, whether to search subdirectories inside each SNID folder (defaults to TRUE)
    """

    for snid in snid_list:
        folder_path = os.path.join(base_path, snid)
        pattern = "**/*.fits" if recursive else "*.fits"
        file_paths = glob.glob(os.path.join(folder_path, pattern), recursive=recursive)

        for file_path in file_paths:
            try:
                
                with fits.open(file_path) as hdul:
                    
                    header = hdul[0].header
                    phot_mode = header.get("PHOTMODE")
                    phot_flam = header.get("PHOTFLAM")
                    phot_zpt = header.get("PHOTZPT")
                    phot_plam = header.get("PHOTPLAM")

                    zp_ab = -2.5 * np.log10(phot_flam) - 5 * np.log10(phot_plam) - 2.408

                    info = {
                        "SNID": snid,
                        "Filename": file_path,
                        "Date": header.get("DATE"),
                        "Photometric Calibration": phot_mode,
                        "Inverse Sensitivity": phot_flam,
                        "Zeropoint": phot_zpt,
                        "Wavelength": phot_plam,
                        "Zeropoint AB": zp_ab
                    }

                    data_list.append(info)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    # Convert to DataFrame and export
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    print(f"CSV exported: {output_csv}")

def get_snid_folders(base_path="source_extractor"):
    """
    Returns a list of folder names (SNIDs) under the base path (base path defaults to source_extractor)
    """
    return [p.name for p in Path(base_path).iterdir() if p.is_dir()]



if __name__=="__main__":
    main()