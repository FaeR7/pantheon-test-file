from astroquery.mast import Observations
import pandas as pd
import subprocess

def main():

    subprocess.call("cd ..", shell = True)

    df = pd.read_csv('data_files/unmatched_pantheon.csv')
    valid_points = []
    def record_index_SP1A(resolved_coord_index):
        """
        If there are images for the supernovae in HST at the resolved query, create a file that stores the index point of the list
        Parameters:
            resolved_coord_index: integer, the index number of the query
        Returns:
            An array with the index points of the list
        """
        print(resolved_coord_index, df['resolved_coord'][resolved_coord_index])

        #try the the query, if a error is thrown then print "No Data Points" and "skip" this data point, else make folder and store data
        try:
            obs_table = Observations.query_criteria(coordinates=df['resolved_coord'][resolved_coord_index],
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
            info = {
                        "SNID": df['SNID'][resolved_coord_index],
                        "RA": df['RA'][resolved_coord_index],
                        "Dec": df['Dec'][resolved_coord_index]
                    }
            valid_points.append(info)
    
    #Goes through each of the resolved coordinates and finds if there is data, then makes valid_points_index_list.csv
    for i in range(len(df['resolved_coord'])):
        record_index_SP1A(i)
        if (i%10 == 0):
            df_vp = pd.DataFrame(valid_points)
            df_vp.to_csv('valid_points_index_list_HST.csv', index=False)

if __name__== "__main__":
    main()