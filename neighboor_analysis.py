## importation
import pandas as pd
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool

def check_neighboor(cell_id, cell_to_location, radius):
    """
    [DEPRECATED]
    use directly the cell corrdinates and a radius to found neighboor,
    not really good results (self detection + duplication of radius value from
    the graph generation part)
    """

    ## extract coordinates
    coord = cell_to_location[cell_id]
    coord = coord.split("_")
    x = float(coord[0])
    y = float(coord[1])
    voisins = []

    ## loop over cells
    for cell_id in list(cell_to_location.keys()):

        #-> extract target coordinates
        t_coord = cell_to_location[cell_id]
        t_coord = t_coord.split("_")
        t_x = float(t_coord[0])
        t_y = float(t_coord[1])

        #-> just check as if its a stupid square for now
        if(abs(t_x-x) <= radius or abs(t_y-y) <= radius):
            voisins.append(cell_id)

    ## return list of cell in range of radius
    return voisins


def check_neighboor_edge_based(cell_id, edge_data):
    """
    An alternative method to fetch cell neighboor, instead
    of raw radius & coordinates distance computation, looks
    for connected cell through the edge file
    """

    
    # look for connected nodes
    voisins = []
    for index, row in edge_data.iterrows():
        if(cell_id == row["source"]):
            voisins.append(row["target"])

    # return list of cell connected to the cell_id
    return voisins

def get_voisin_count_as_percentage(cluster_to_voisins_to_count):
    """
    """

    ## parameters
    cluster_to_voisins_to_percentage = {}

    ## compute presentages
    for cluster in list(cluster_to_voisins_to_count.keys()):
        cluster_to_voisins_to_percentage[cluster] = {}
        voisinage = cluster_to_voisins_to_count[cluster]

        ## get total of count in voisinage
        total = 0
        for voisin in list(voisinage.keys()):
            total+= voisinage[voisin]

        ## compute percentage
        for voisin in list(voisinage.keys()):
            scalar = (float(voisinage[voisin]) / float(total))*100.0
            cluster_to_voisins_to_percentage[cluster][voisin] = scalar

    ## return dictionnary
    return cluster_to_voisins_to_percentage



def get_voisin_count_as_percentage_mean_distribution_mode(cluster_to_voisins_to_percentage_list):
    """
    used in run_on_multiple_files function to agregate the voisins distribution of each data file
    and consider the mean of the distribution (not the distribution on the aggregate count) to be plot
    on the heatmap
    
    Yeah, biologist requirement, sort of
    """
    
    # parameters
    cluster_to_voisins_to_mean_percentage = {}

    # loop and compute
    for cluster_to_voisins_to_percentage in cluster_to_voisins_to_percentage_list:
        for c1 in cluster_to_voisins_to_percentage.keys():
            cluster_to_voisins_to_mean_percentage[c1] = {}
            for c2 in cluster_to_voisins_to_percentage[c1].keys():
                percentage_list = cluster_to_voisins_to_percentage[c1][c2]
                cluster_to_voisins_to_mean_percentage[c1][c2] = np.mean(percentage_list)

    # return computed information
    return cluster_to_voisins_to_mean_percentage







def craft_heatmap(cluster_to_voisins_to_percentage, save_file_name):
    """
    """

    ## parameters
    matrix = []

    ## craft sort cluster list
    cluster_list = list(cluster_to_voisins_to_percentage.keys())
    cluster_list.sort()

    ## craft matrix
    for cluster in cluster_list:
        vector = []
        data = cluster_to_voisins_to_percentage[cluster]
        for voisin in cluster_list:
            if(voisin in data.keys()):
                vector.append(data[voisin])
            else:
                vector.append(0)
        matrix.append(vector)

    ## craft figure
    ax = sns.heatmap(matrix, xticklabels=cluster_list, yticklabels=cluster_list, cmap='viridis')
    plt.title("Neighborhood Matrix")
    plt.xlabel("Neighborhood Distribution (% of clusters)")
    plt.ylabel("Cluster")
    plt.savefig(save_file_name)
    plt.close()






def run(data_file, edge_file):
    """
    """

    ## parameters
    id_to_coordinates = {}
    cluster_to_id_list = {}
    id_to_cluster = {}
    cluster_to_voisins_to_count = {}
    radius = 1
    heatmap_image_file = "heatmap_test.png"

    ## load data_file
    df = pd.read_csv(data_file)

    ## get cell to location
    for index, row in df.iterrows():

        #-> get id
        if(" Cell_ID" in row.keys()):
            cell_id = row[" Cell_ID"]
        elif("Cell_ID" in row.keys()):
            cell_id = row["Cell_ID"]
        else:
            cell_id = f"cell_{index}"

        #-> get coordinates
        coord = str(row["centroid_X"])+"_"+str(row["centroid_Y"])

        #-> get label
        if("Cluster" in row.keys()):
            label = row["Cluster"]
        elif("pgraph" in row.keys()):
            label = row["pgraph"]

        #-> update id to coordinates
        id_to_coordinates[cell_id] = coord

        #-> update id to cluster
        id_to_cluster[cell_id] = label

        #-> update cluster to id
        if(label in list(cluster_to_id_list)):
            cluster_to_id_list[label].append(cell_id)
        else:
            cluster_to_id_list[label] = [cell_id]

    ## loop over types
    for label in list(cluster_to_id_list.keys()):

        cluster_to_voisins_to_count[label] = {}

        #-> for cell in types get close cells
        for cell_id in cluster_to_id_list[label]:

            #-> get voisinage label enrichement
            voisinage = check_neighboor_edge_based(cell_id, edge_file)

            #-> get type of close cell & update cell_type_to_prox_cell_type
            for cid in voisinage:
                cluster = id_to_cluster[cid]
                if(cluster in cluster_to_voisins_to_count[label].keys()):
                    cluster_to_voisins_to_count[label][cluster] +=1
                else:
                    cluster_to_voisins_to_count[label][cluster] = 1


    ## work with cell_type_to_prox_cell_type
    cluster_to_voisins_to_percentage = get_voisin_count_as_percentage(cluster_to_voisins_to_count)

    ## craft heatmap
    craft_heatmap(cluster_to_voisins_to_percentage, heatmap_image_file)








def run_on_multiple_files(data_file_list, heatmap_image_file, graph_folder):
    """

    - data_file_list :  list of data file
    - heatmap_image_file :  name of the generated heatmap
    - graph_folder :    emplacement of the graph folder, should contain nodes and edges subfolder
                        (only looking for the edges here), edge file name should be the same at data file name 
    

    => trouble with the computation of percentages, at this point we are trying two different methods:
        - total_count : gather all voisins for each cluster from all files and compute distribution
        - mean_count : for each dataset compute the distribution, then take the mean of the distribution accross the data file
    """

    ## parameters
    id_to_coordinates = {}
    cluster_to_id_list = {}
    id_to_cluster = {}
    cluster_to_voisins_to_count = {}
    cluster_to_voisins_to_percentage_list = []
    shutup = False
    distribution_mode = "mean_count"
    multi_processing_mode = False

    # init log file
    log_file = heatmap_image_file.split(".")
    log_file = log_file[0]+".log"
    log_data = open(log_file, "w")

    # loop over data file in data list file
    cmpt = 0
    for data_file in data_file_list:

        ## load data_file
        df = pd.read_csv(data_file)

        # look for corresponding edge file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        edge_file = data_file.split("/")
        edge_file = edge_file[-1]
        edge_file = f"{graph_folder}/edges/{edge_file}"
        if(not os.path.isfile(edge_file)):
            print(f"[{current_time}]<<!>> can't find edge file {edge_file}")
            log_data.write(f"[current_time]<<!>> can't find edge file {edge_file}")
            log_data.close()
            return 0
        edge_data = pd.read_csv(edge_file)
        log_data.write(f"[current_timeassigning edge file {edge_file} to data file {data_file}\n")
       
        ## get cell to location
        for index, row in df.iterrows():

            #-> get id
            if(" Cell_ID" in row.keys()):
                cell_id = row[" Cell_ID"]
            elif("Cell_ID" in row.keys()):
                cell_id = row["Cell_ID"]
            else:
                cell_id = f"cell_{index}"

            #-> get coordinates
            coord = str(row["centroid_X"])+"_"+str(row["centroid_Y"])

            #-> get label
            if("Cluster" in row.keys()):
                label = row["Cluster"]
            elif("pgraph" in row.keys()):
                label = row["pgraph"]

            #-> update id to coordinates
            id_to_coordinates[cell_id] = coord

            #-> update id to cluster
            id_to_cluster[cell_id] = label

            #-> update cluster to id
            if(label in list(cluster_to_id_list)):
                cluster_to_id_list[label].append(cell_id)
            else:
                cluster_to_id_list[label] = [cell_id]
        
        ## loop over types
        for label in list(cluster_to_id_list.keys()):

            if(label not in cluster_to_voisins_to_count.keys()):
                cluster_to_voisins_to_count[label] = {}
               
            # -> classic
            if(not multi_processing_mode):
                
                #-> for cell in types get close cells
                for cell_id in cluster_to_id_list[label]:

                    #-> get voisinage label enrichement
                    voisinage = check_neighboor_edge_based(cell_id, edge_data)

                    #-> get type of close cell & update cell_type_to_prox_cell_type
                    for cid in voisinage:
                        if(cid in id_to_cluster.keys()):
                            cluster = id_to_cluster[cid]
                            if(cluster in cluster_to_voisins_to_count[label].keys()):
                                cluster_to_voisins_to_count[label][cluster] +=1
                            else:
                                cluster_to_voisins_to_count[label][cluster] = 1

                        else:
                            now = datetime.now()
                            current_time = now.strftime("%H:%M:%S")
                            log_data.write(f"[{current_time}]<<!>> can't find cluster for {cell_id}, look your data file\n")
            else:
                pass
        
        if(distribution_mode == "mean_count"):

            # compute voisinage distribution for this specific file
            cluster_to_voisins_to_percentage = get_voisin_count_as_percentage(cluster_to_voisins_to_count)
            
            # add values in list of percentages
            cluster_to_voisins_to_percentage_list.append(cluster_to_voisins_to_percentage)

            # reinit cluster_to_voisins_to_count for the next file
            cluster_to_voisins_to_count = {}

        # display progress
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if(not shutup):
            cmpt+=1
            progress = (float(cmpt) / len(data_file_list))*100.0
            print(f"[{current_time}][PROGRESS] {cmpt} / {len(data_file_list)} >> {progress} %")
        log_data.write(f"[{current_time}][PROGRESS] {cmpt} / {len(data_file_list)} >> {progress} %\n")

    # close log file
    log_data.close()
    
    ## work with cell_type_to_prox_cell_type
    if(distribution_mode == "total_count"):
        cluster_to_voisins_to_percentage = get_voisin_count_as_percentage(cluster_to_voisins_to_count)
    elif(distribution_mode == "mean_count"):
        cluster_to_voisins_to_percentage = get_voisin_count_as_percentage_mean_distribution_mode(cluster_to_voisins_to_percentage_list)
    
    ## craft heatmap
    craft_heatmap(cluster_to_voisins_to_percentage, heatmap_image_file)



if __name__ == "__main__":
    """
    Used as test environnement
    """

    # importation
    import glob

    # parameters
    # -> small & real dataset    
    # big_test_list_file = glob.glob("/home/bran/Workspace/misc/hypernet_voisin_test/data/*.csv")
    # heatmap_image_file = "neighboor_analysis/tests/heatmap.png"
    # graph_folder_name = "/home/bran/Workspace/misc/hypernet_voisin_test/graph"
    
    # -> toy dataset
    big_test_list_file = glob.glob("/home/bran/Workspace/misc/graph_test/raw_data/*.csv")
    heatmap_image_file = "/home/bran/Workspace/HYPERNET/neighboor_analysis/tests/toy_heatmap_mean.png"
    graph_folder_name = "/home/bran/Workspace/misc/graph_test/graph"

    # -> tiny toy dataset
    # big_test_list_file = glob.glob("/home/bran/Workspace/HYPERNET/neighboor_analysis/tests/*.csv")
    # heatmap_image_file = "/home/bran/Workspace/HYPERNET/neighboor_analysis/tests/tiny_toy_heatmap.png"
    # graph_folder_name = "/home/bran/Workspace/HYPERNET/neighboor_analysis/tests/graph"
    
    # go for test run
    run_on_multiple_files(
        big_test_list_file,
        heatmap_image_file,
        graph_folder_name
    )
