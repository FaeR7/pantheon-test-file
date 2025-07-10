import sys, os, glob
import argparse
import pandas as pd
import numpy as np

def define_args():
    parser = argparse.ArgumentParser(description="Generate DS9 region files", conflict_handler='resolve')
    
    parser.add_argument(
        "supernova_names", 
        type=str, 
        nargs="+", 
        help="List of supernova names to process."
    )

    parser.add_argument(
        "--fits_summary_path", 
        type=str, 
        default="./fits_summary.csv", 
        help="Path to the FITS summary CSV file."
    )

    parser.add_argument(
        "--brightest_galaxy_path", 
        type=str, 
        default="./brightest_galaxy.csv", 
        help="Path to the brightest galaxy CSV file."
    )

    parser.add_argument(
        "--all_output_filename", 
        type=str, 
        default="ds9_cmds_all.reg", 
        help="Filename for the combined region file output."
    )

    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="./region_files", 
        help="Directory where individual region files will be saved."
    )

    return parser.parse_args()


# ds9:
#ellipse(186.8194200,9.4313379,10.000",30.000",44.999994) # color=red width=4
#ellipse(RA, Dec, minor axis lenght, major axis length, Poisition Angle PA in deg)
def make_region_command_for_position(ra,dec,position_angle,major_axis,minor_axis,name=None,symbol='circle',color='red',width=4,size='10"'):
    print(f'Making region command: RA_Galaxy {ra}, Dec_Galaxy {dec}, name {name}, Position_Angle {position_angle}, Major_Axis {major_axis}, Minor_Axis {minor_axis}')
    name='{'+f'{name}'+'}'
    s = f'circle({ra},{dec},{size},{position_angle},{major_axis},{minor_axis}) # color={color} width={width} text={name}'
    print(s)
    return(s)

def get_region_info(cd_matrix, cov_matrix):
    # 3. World-space covariance matrix (deg²)
    cov_world = cd_matrix @ cov_matrix @ cd_matrix.T

    # 4. Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_world)  # always returns sorted values

    # Convert eigenvalues (deg²) to axis lengths (arcsec)
    # Note: 1 deg = 3600 arcsec, so variance in deg² -> stddev in arcsec = sqrt(val) * 3600
    a_arcsec = 3600 * np.sqrt(eigvals[1])  # major axis
    b_arcsec = 3600 * np.sqrt(eigvals[0])  # minor axis

    # 5. Position angle (in degrees, from North to East)
    vec = eigvecs[:, 1]  # eigenvector of major axis
    theta_rad = np.arctan2(vec[0], vec[1])  # careful: note the order (RA, Dec)
    theta_deg = np.degrees(theta_rad)

    # Normalize angle to 0–180
    theta_deg = (theta_deg + 360) % 180

    return theta_deg, a_arcsec, b_arcsec

def save_region_file(filepath, output):
    print(f'Writing region file to {filepath}')
    with open(filepath, "w") as file:
        for line in output:
            file.write(str(line) + "\n")
    file.close()

def combine_fs_and_bg(supernova_names, fits_summary, brightest_galaxy):
    for sn_name in supernova_names:
            # print("SN name: ",sn_name)

            sn_ix_bg = brightest_galaxy[brightest_galaxy["SNID"] == sn_name].index.values
            # print("Brightest galaxy indices: ",sn_ix_bg)

            sn_ix = np.where(fits_summary["SNID"] == sn_name)[0]
            # print("Fits summary indices: ",sn_ix)

            for index in sn_ix_bg:
                filename = brightest_galaxy.loc[index,'File_key']
                filename_ix_fs = sn_ix[np.where(fits_summary.loc[sn_ix, "filenameshort"] == filename+".fits")[0]]

                if len(filename_ix_fs)>1:
                    raise RuntimeError(f"filename_ix_fs is returning multiple matches for filenameshort in fits_summary: {filename_ix_fs}")

                for column_name in brightest_galaxy.columns:
                    if column_name not in fits_summary.columns:
                        fits_summary[column_name] = np.nan
                    fits_summary.at[filename_ix_fs[0], column_name] = brightest_galaxy.at[index, column_name]

    return(fits_summary)

def get_region_command(index, fits_summary):
    print("Index:", index)
    print("Row: \n",fits_summary.iloc[[index]].to_string())

    file_key = fits_summary.at[index, "File_key"]
    if isinstance(file_key, float) and np.isnan(file_key):
        return None

    if ra_galaxy is None or dec_galaxy is None:
        ra_galaxy = fits_summary.at[index, "RA_Galaxy"]
        dec_galaxy = fits_summary.at[index, "Dec_Galaxy"]
        print(f"Setting RA and Dec: {ra_galaxy}, {dec_galaxy}")
    
    print(f"RA and Dec: {ra_galaxy}, {dec_galaxy}")
    
    # 1. Pixel-space covariance matrix
    cov_matrix = np.array([
        [fits_summary.at[index, "CXX"], fits_summary.at[index, "CXY"]],
        [fits_summary.at[index, "CXY"], fits_summary.at[index, "CYY"]]
    ])

    # 2. CD matrix (deg/pixel)
    cd_matrix = np.array([
        [fits_summary.at[index, "CD1_1"], fits_summary.at[index, "CD1_2"]],
        [fits_summary.at[index, "CD2_1"], fits_summary.at[index, "CD2_2"]]
    ])

    theta_deg, a_arcsec, b_arcsec = get_region_info(cd_matrix, cov_matrix)

    cmd = make_region_command_for_position(ra_galaxy, dec_galaxy, theta_deg, a_arcsec, b_arcsec, name=sn_name)

    return cmd

if __name__ == "__main__":
    args = define_args()
    
    # pandas reads the table at the path args.fits_summary_path and then returns it to the variable fits_summary
    fits_summary = pd.read_table(args.fits_summary_path,sep=',')
    print(fits_summary.to_string())

    # same thing -- reading table at args.brightest_galaxy_path and assigning it to brightest_galaxy variable
    brightest_galaxy = pd.read_table(args.brightest_galaxy_path,sep=',')
    print(brightest_galaxy.to_string())

    pd.set_option('display.max_colwidth', None)
    
    # receiving end: the "filenameshort" column in the fits_summary table
    fits_summary['filenameshort'] = fits_summary['filename'].str.replace('.*\/','',regex=True)

    # combines fits_summary and brightest_galaxy tables
    print("Combining fits_summary and brightest_galaxy tables...")
    fits_summary = combine_fs_and_bg(args.supernova_names, fits_summary, brightest_galaxy)
    print(fits_summary.head().to_string())
    fits_summary.to_csv(f"./fits_summary_combined.csv")

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    for sn_name in args.supernova_names:
        print("SN name: ",sn_name)

        sn_ix = np.where(fits_summary["SNID"] == sn_name)[0]

        ra_galaxy = None
        dec_galaxy = None

        all_output = ['global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1']
        all_output.append('fk5')
        
        for index in sn_ix:
            cmd = get_region_command(index, fits_summary)

            # make region file
            output = ['global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1']
            output.append('fk5')
            output.append(cmd)
            all_output.append(cmd)

            filepath = f"{args.out_dir}/{fits_summary.at[index, 'File_key']}_region_file.reg"
            save_region_file(filepath, output)
    
    filepath = f"./{args.out_dir}/{args.all_output_filename}"
    save_region_file(filepath, all_output)