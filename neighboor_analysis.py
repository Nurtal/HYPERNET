## importation
import pandas as pd
import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def check_neighboor(cell_id, cell_to_location, radius):
    """
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
    ax = sns.heatmap(matrix, xticklabels=cluster_list, yticklabels=cluster_list)
    plt.title("Neighborhood Matrix")
    plt.xlabel("Neighborhood Distribution (% of clusters)")
    plt.ylabel("Cluster")
    plt.savefig(save_file_name)
    plt.close()






def run(data_file):
    """
    """

    ## parameters
    id_to_coordinates = {}
    cluster_to_id_list = {}
    id_to_cluster = {}
    cluster_to_voisins_to_count = {}
    radius = 0.2
    heatmap_image_file = "heatmap_test.png"

    ## load data_file
    df = pd.read_csv(data_file)

    ## get cell to location
    for index, row in df.iterrows():

        #-> get id
        cell_id = row[" Cell_ID"]

        #-> get coordinates
        coord = str(row["centroid_X"])+"_"+str(row["centroid_Y"])

        #-> get label
        label = row["Cluster"]

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
            voisinage = check_neighboor(cell_id, id_to_coordinates, radius)

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




def run_on_multiple_files(data_file_list, heatmap_image_file, radius):
    """
    """

    ## parameters
    id_to_coordinates = {}
    cluster_to_id_list = {}
    id_to_cluster = {}
    cluster_to_voisins_to_count = {}


    for data_file in data_file_list:

        ## load data_file
        df = pd.read_csv(data_file)

        ## get cell to location
        for index, row in df.iterrows():

            #-> get id
            if(" Cell_ID" in row.keys()):
                cell_id = row[" Cell_ID"]
            else:
                cell_id = index

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

            #-> for cell in types get close cells
            for cell_id in cluster_to_id_list[label]:

                #-> get voisinage label enrichement
                voisinage = check_neighboor(cell_id, id_to_coordinates, radius)

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



