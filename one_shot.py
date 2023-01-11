"""
single file to run hypernet as a module
"""

# importation
import os
import pandas as pd
import numpy as np


#---------------------#
# INTERNAL FUNNCTIONS #
#---------------------#
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


def simple_display(edge_file, node_file, pos_file, output_file_name):
    """
    """

    ## craft dataframe
    #-> load edge_file & craft edge list
    df_edge = pd.read_csv(edge_file)
    edge_list = []
    for index, row in df_edge.iterrows():
        source = row['source']
        target = row['target']
        edge_list.append((source,target))

    #-> load nodes
    df_node =pd.read_csv(node_file)
    node_list = []
    cluster_to_node = {}

    #-> patch cluster column name if needed
    if("pgraph" in df_node.keys()):
        df_node = df_node.rename(columns={"pgraph":"cluster"})

    for index, row in df_node.iterrows():

        #-> extract id & properties
        if('ID' in row.keys()):
            cell_id = row["ID"]
        else:
            cell_id = "cell_"+str(index+1)
        properties = {}
        for k in row.keys():
            if(k != "ID"):
                properties[k] = row[k]
        node_list.append((cell_id,properties))

        ## extract clusters (WARING cluster is present in list of rows)
        if(row["cluster"] not in cluster_to_node.keys()):
            cluster_to_node[row["cluster"]] = [(cell_id)]
        else:
            cluster_to_node[row["cluster"]].append((cell_id))

    ## craft network
    #-> init graph
    G = nx.Graph()
    #-> add nodes
    G.add_nodes_from(node_list)
    #-> add edges
    G.add_edges_from(edge_list)

    ## display network
    #-> extract position
    pos = {}
    df_pos = pd.read_csv(pos_file)
    i = 1
    for index, row in df_pos.iterrows():
        x = float(row["centroid_X"])
        y = float(row["centroid_Y"])
        pos["cell_"+str(i)] = np.array([x,y])
        i+=1

    #-> draw the nodes for each cluster
    color = cm.rainbow(np.linspace(0, 1, len(list(cluster_to_node.keys()))))
    i = 0
    for cluster in cluster_to_node.keys():

        #--> extract node_list
        node_list = cluster_to_node[cluster]

        #--> extract color
        c = color[i]

        #--> draw the nodes
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=c, node_size=3)

        #--> update cmpt
        i+=1

    #-> draw the edge
    nx.draw_networkx_edges(G, pos, edgelist=edge_list)
    #nx.write_gml(G, "test.gml")
    #-> display
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(output_file_name)
    plt.close()


#----------------#
# CORE FUNCTIONS #
#----------------#
def generate_graph(data_file, output_folder):
    """
    """

    # parameters
    neighbour_radius = 4
    data_name = data_file.split("/")
    data_name = data_name[-1]
    
    # init output folder
    if(not os.path.iddir(output_folder)):
        os.mkdir(output_folder)
        os.mkdir(f"{output_folder}/nodes")
        os.mkdir(f"{output_folder}/edges")

    # craft edge dataframe
    df_edge = craft_edge_dataframe(data_file, neighbour_radius, output_folder)

    # craft node dataframe
    df_node = craft_node_dataframe(data_file, output_folder)

    # save nodes and edges
    df_edge.to_csv(f"{output_folder}/edges/edges_{data_name}.csv", index=False)
    df_node.to_csv(f"{output_folder}/nodes/nodes_{data_name}.csv", index=False)


def dislay_graph(data_file, output_folder, output_file_name):
    """
    """

    # parameters
    data_name = data_file.split("/")
    data_name = data_name[-1]
    edge_file = f"{output_folder}/edges/edges_{data_name}.csv"
    node_file = f"{output_folder}/nodes/nodes_{data_name}.csv"

    # call display function
    simple_display(
        edge_file,
        node_file,
        data_file,
        output_file_name
    )


def get_direct_neighboor(cell_id, output_folder):
    """
    """

    # parameters
    data_name = data_file.split("/")
    data_name = data_name[-1]
    edge_file = f"{output_folder}/edges/edges_{data_name}.csv"
    voisins = []

    # get list of neigboor
    df = pd.read_csv(edge_file)
    for index, row in df.iterrows():
        source = row["source"]
        target = row["target"]
        if(cell_id == source and target not in voisins):
            voisins.append(target)
        if(cell_id == target and source not in voisins):
            voisins.append(source)

    # return list of neighboor
    return voisins

