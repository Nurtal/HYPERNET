## importation
import pandas as pd
import numpy as np


def craft_edge_dataframe(data_file, neighbour_radius, output_folder):
    """
    craft and return the edge dataframe

    guide ressources:
        https://stellargraph.readthedocs.io/en/stable/demos/basics/loading-pandas.html
    """


    ## parameters
    node_to_coordinates = {}
    edge_list = []
    node_cmpt = 0
    output_name = data_file.split("/")
    output_name = output_name[-1]
    output_name = output_folder+"/graph/edges/"+str(output_name)

    ## load fcs file
    df = pd.read_csv(data_file)

    ## loop over cell in fcs file
    for index, row in df.iterrows():

        #-> extract coordinates
        x = row["centroid_X"]
        y = row["centroid_Y"]

        #-> update grid information
        node_cmpt +=1
        node_name = "cell_"+str(node_cmpt)
        coordinates = str(int(x))+"_"+str(int(y))
        node_to_coordinates[node_name] = coordinates

    ## loop over n1 in node to coordinates
    for n1 in node_to_coordinates:
        c_array = node_to_coordinates[n1]
        c_array = c_array.split("_")
        x1 = int(c_array[0])
        y1 = int(c_array[1])

        #-> loop over n2 in node to coordinates
        for n2 in node_to_coordinates:
            c_array = node_to_coordinates[n2]
            c_array = c_array.split("_")
            x2 = int(c_array[0])
            y2 = int(c_array[1])

            #--> compute euclidien distance between n1 and n2
            pt1 = np.array([x1,y1])
            pt2 = np.array([x2,y2])
            dist = np.linalg.norm(pt1 - pt2)

            #--> test distance with treshold, create an edge ie its below
            if(dist < neighbour_radius and n1 != n2):
                if([n2,n1] not in edge_list):
                    edge_list.append([n1,n2])

    ## craft datafrale from edge list
    df = pd.DataFrame(edge_list, columns = ['source', 'target'])

    ## save dataframe
    df.to_csv(output_name, index=False)





def craft_node_dataframe(data_file, output_folder):
    """
    """

    ## parameters
    node_cmpt = 0
    header = ["ID"]
    data = []
    output_name = data_file.split("/")
    output_name = output_name[-1]
    output_name = output_folder+"/graph/nodes/"+str(output_name)

    ## load dataframe
    df = pd.read_csv(data_file)

    ## loop over dataframe
    for index, row in df.iterrows():

        #-> get header
        for k in list(row.keys()):
            if(k not in ["centroid_X","centroid_Y"] and k not in header):
                header.append(k)

        #-> get cell id
        node_cmpt +=1
        node_name = "cell_"+str(node_cmpt)

        #-> craft line
        line = [node_name]
        for k in list(row.keys()):
            if(k not in ["centroid_X","centroid_Y"]):
                line.append(row[k])

        #-> add line to data
        data.append(line)

    ## craft dataframe from list of vector
    df = pd.DataFrame(data, columns = header)
    df = df.set_index("ID")

    ## save dataframe
    df.to_csv(output_name, index=False)

    ## return dataframe
    return df
