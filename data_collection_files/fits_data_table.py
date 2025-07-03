import pandas as pd 
from astropy.io import fits
import glob
import os
from pathlib import Path
import subprocess

def main():

    subprocess.call("cd ..", shell = True)

    snids = get_snid_folders()
    print(len(snids))
    extract_fits_by_snid(snids, output_csv="fits_summary.csv", recursive=True)


def extract_fits_by_snid(snid_list, base_path="pantheon_data_folder", output_csv="fits_summary.csv", recursive=True, data_list = []):
    """
    Loops over SNIDs, reads .fits files in /pantheon/{SNID}/ folders
        - Extracts the Telescope, Instrument Used, and Filter
    Parameters:
    - snid_list: list of str, SNID folder names
    - base_path: str, base directory containing SNID subfolders (defaults to pantheon_data_folder)
    - output_csv: str, name of output CSV file (defaults to fits_summary.csv)
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
                    header_2 = hdul[1].header

                    date = header.get("DATE")
                    detector = header.get("INSTRUME")
                    filter = header.get("FILTER")


                    info = {
                        "SNID": snid,
                        "filename": file_path,
                        "Date": date,
                        "Telescope": header.get("TELESCOP"),
                        "Instrument": detector,
                        "Filter": filter,
                        "CD1_1": header_2.get("CD1_1"),
                        "CD1_2": header_2.get("CD1_2"),
                        "CD2_1": header_2.get("CD2_1"),
                        "CD2_2": header_2.get("CD2_2")
                    }

                    data_list.append(info)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Convert to DataFrame and export
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    print(f"CSV exported: {output_csv}")

def get_snid_folders(base_path="pantheon_data_folder"):
    """
    Returns a list of folder names (SNIDs) under the base path (base path defaults to pantheon_data_folder)
    """
    return [p.name for p in Path(base_path).iterdir() if p.is_dir()]



if __name__=="__main__":
    main()