## imporation
import shutil
import os
import numpy as np
import glob


def load_raw_dataset(label_to_file, output_folder):
    """
    """

    ## paramaters
    file_list = []

    ## create file list
    for k in label_to_file:
        flist = label_to_file[k]
        for f in flist:
            if(f not in file_list):
                file_list.append(f)

    ## create raw sub folder
    if(not os.path.isdir(output_folder+"/raw_data")):
        os.mkdir(output_folder+"/raw_data")

    ## copy all files into raw sub folder
    for f in file_list:
        destination = f.split("/")
        destination = destination[-1]
        destination = output_folder+"/raw_data/"+destination
        shutil.copy(f, destination)




def normalize_dataset(output_folder):
    """
    IN PROGRESS
    """

    ## importation
    import pandas as pd

    ## parameters
    marker_to_scalar = {}
    marker_to_mean = {}
    marker_to_std = {}
    marker_list = []

    ## create normalized sub folder
    if(not os.path.isdir(output_folder+"/normalized_data")):
        os.mkdir(output_folder+"/normalized_data")

    ## identify target files & markers
    target_files = glob.glob(output_folder+"/raw_data/*.csv")
    df = pd.read_csv(target_files[0])
    for k in list(df.keys()):
        if(k not in ["centroid_X", "centroid_Y"]):
            marker_list.append(k)
            marker_to_scalar[k] = []

    ## Part 1 -  get the mean and std for all markers
    for tf in target_files:

        #-> load data
        df = pd.read_csv(tf)

        #-> loop over data & get scalars for each markers
        for index, row in df.iterrows():
            for k in list(row.keys()):
                if(k in marker_list):
                    marker_to_scalar[k].append(row[k])

    ## compute mean and std
    for k in marker_to_scalar.keys():
        vector = marker_to_scalar[k]
        marker_to_mean[k] = np.mean(vector)
        marker_to_std[k] = np.std(vector)

    ## Part 2 - Apply standardization
    ## loop over fcs file
    for tf in target_files:

        #-> load dataframe
        df = pd.read_csv(tf)

        #-> apply standardization
        for marker in marker_list:
            df[marker] = ((df[marker] - marker_to_mean[marker]) / marker_to_std[marker])

        #-> save dataframe to normalize file
        output_name = tf.replace(".csv", "_normalized.csv")
        output_name = output_name.replace("raw_data", "normalized_data")
        df.to_csv(output_name, index=False)




def simple_discretization(output_folder):
    """
    """

    ## importation
    import pandas as pd
    import glob

    ## create normalized sub folder
    if(not os.path.isdir(output_folder+"/discretized_data")):
        os.mkdir(output_folder+"/discretized_data")

    ## loop over target files
    for tf in glob.glob(output_folder+"/normalized_data/*_normalized.csv"):

        #-> init new matrix
        matrix = []

        #-> load target files
        df = pd.read_csv(tf)

        #-> get header
        header = list(df.keys())

        #-> discretize
        for index, row in df.iterrows():
            vector = []
            for k in list(row.keys()):
                if(k in ["centroid_X", "centroid_Y"]):
                    vector.append(row[k])
                else:
                    scalar = row[k]
                    new_scalar = "NA"

                    #--> discretize
                    if(scalar < 0.2):
                        new_scalar = 0
                    elif(scalar < 0.4):
                        new_scalar = 1
                    elif(scalar < 0.6):
                        new_scalar = 2
                    elif(scalar < 0.8):
                        new_scalar = 3
                    elif(scalar <= 1):
                        new_scalar = 4
                    elif(scalar <= 2):
                        new_scalar = 5
                    else:
                        new_scalar = 6

                    # update vector
                    vector.append(new_scalar)

            # update matrix
            matrix.append(vector)

        ## craft and save csv
        df = pd.DataFrame(matrix, columns=header)
        output_name = tf.replace(".csv", "_discretized.csv")
        output_name = output_name.replace("normalized_data", "discretized_data")
        df.to_csv(output_name, index=False)
