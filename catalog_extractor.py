import subprocess
import os
from astropy.io import fits
import pandas as pd
import shutil

def source_extractor(source_extractor_path = ""):
    """
    Params:
        - source_extractor_path: the path to where the source extractors files are located
    """
    if not os.path.exists(source_extractor_path + "default.conv"):
        print("HEY MAKE A DEFAULT CONV FILE")

    if not os.path.exists(source_extractor_path + "default.param"):
        subprocess.call("sex -dp -> default.param", shell=True)
        print("HEY CHANGE THE DEFAULT PARAMETER FILE")
    
    if not os.path.exists(source_extractor_path + "default.sex"):
        subprocess.call("sex -d -> default.sex", shell=True)
        print("HEY MAKE THE DEFAULT SEX FILE")

def catalog_creator(SNID, filename, filter, home_path, source_path = ""):
    if filter == 'detection':
        return;

    base_file = os.path.basename(filename)
    base, _ = os.path.splitext(base_file)

    done = check_file_exist(home_path, source_path, SNID, base)
    if done == True:
        return;

    dest_path = home_path + "/" + filename

    exec_string = 'sex ' + dest_path + ' -c ' + 'default.sex'

    cat_file_name = "test.cat"
    new_cat_file_name = f"{base}.cat"

    fits_file_name = "check.fits"
    new_fits_file_name = f"{base}.fits"

    
    subprocess.call(exec_string, shell = True);
    os.rename(cat_file_name, new_cat_file_name)
    os.rename(fits_file_name, new_fits_file_name)
    move_files_group(SNID, base, new_cat_file_name, new_fits_file_name, home_path, source_path)

def check_file_exist(home_path, source_path, SNID, base):
    """
    Makes the process able to be stopped and started at your discretion
    Returns:
        Boolean value of if the file exists
    """
    return os.path.exists(f"{home_path}/{source_path}/{SNID}/{base}")
    

        

def move_files_group(SNid, base, cat_file_name, fits_file_name, home_path, source_path = ''):
    """
    Moves files from one location to another given the filename, home_path, and source path
    Parameters:
        SNid: The SNID of the suoernova
        file_name: the name of the file
        home_path: the path to the file (generally the original home_path)
        source_path: the path of the sub folder that should store files (defaults to no folder but causes problems)
    """
    try:
        destination_path = f"{home_path}/{source_path}/{SNid}/{base}"
        os.makedirs(destination_path, exist_ok=True)
        shutil.move(cat_file_name, destination_path)
        shutil.move(fits_file_name, destination_path)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




def main():
    df = pd.read_csv('data_files/fits_summary.csv')
    home_dir = os.getcwd()

    #activates the conda enviroment
    conda_act = "conda activate " + home_dir + "/.conda"
    subprocess.call("conda init", shell = True)
    subprocess.call(conda_act, shell=True)

    #Makes sure that each of the source extractor files are avaliable and found
    source_extractor()
    for row in df.itertuples():
        catalog_creator(row.SNID, row.filename, row.Filter, home_dir, source_path= "source_extractor")

if __name__=="__main__":
    main()
                         

