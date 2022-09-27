## importation
import phenograph
import numpy as np
import pandas as pd




def annotation_with_pehnograph(file_list, variable_to_keep, output_folder):
    """
    """

    ## parameters
    df_list = []

    ## loop over file list & create a giant dataframe
    cmpt = 0
    for data_file in file_list:

        #-> load dataframe
        df = pd.read_csv(data_file)

        #-> extract coordinates
        variables = ["centroid_X","centroid_Y"]
        for var in variable_to_keep:
            variables.append(var)

        #-> select variable_to_keep
        df = df[variables]

        #-> create tag_id for the cell
        df["FILE_ID"] = cmpt
        cmpt+=1

        ## upddate dataframe list
        df_list.append(df)


    ## fusion dataframe
    df = pd.concat(df_list)

    ## convert to np array
    data = df[variable_to_keep]
    data = np.array(data)

    ## run phenopgraph
    communities, graph, Q = phenograph.cluster(data)

    ## get cell coordinates to cluster
    file_to_cell_to_cluster = {}
    cmpt = 0
    for index, row in df.iterrows():
        f = file_list[int(row["FILE_ID"])]
        x = row["centroid_X"]
        y = row["centroid_Y"]
        cluster = communities[cmpt]
        cmpt+=1


        if(f not in file_to_cell_to_cluster.keys()):
            file_to_cell_to_cluster[f] = {}
            file_to_cell_to_cluster[f][str(x)+"_"+str(y)] = cluster
        else:
            file_to_cell_to_cluster[f][str(x)+"_"+str(y)] = cluster



    ## save results
    for f in file_to_cell_to_cluster.keys():
        data = file_to_cell_to_cluster[f]
        f_out = f.split("/")
        f_out = f_out[-1]
        f_out = output_folder+"/discretized_data/"+str(f_out)
        f_out = f_out.replace(".csv", "_phenograph_cluster.csv")
        output_data = open(f_out, "w")
        output_data.write("centroid_X,centroid_Y,cluster\n")
        for coordinates in data.keys():
            coord = coordinates.split("_")
            x = coord[0]
            y = coord[1]
            cluster = file_to_cell_to_cluster[f][coordinates]
            output_data.write(str(x)+","+str(y)+","+str(cluster)+"\n")
        output_data.close()
