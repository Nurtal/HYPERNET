## importation
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np



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

    #-> display
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(output_file_name)
    plt.close()





"""
## real data
edge_file = "/home/bran/Workspace/misc/hypernet_test4/graph/edges/slide11roi10_1_mean_phenograph_cluster.csv"
node_file = "/home/bran/Workspace/misc/hypernet_test4/graph/nodes/slide11roi10_1_mean_phenograph_cluster.csv"
pos_file = "/home/bran/Workspace/misc/hypernet_test4/discretized_data/slide11roi10_1_mean_phenograph_cluster.csv"
output_name = "/home/bran/Workspace/misc/hypernet_test4/graph/graph.svg"
"""

"""
## test data
edge_file = "/home/bran/Workspace/misc/hypernet_test4/graph/edges/test_edges_filtered.csv"
node_file = "/home/bran/Workspace/misc/hypernet_test4/graph/nodes/test_nodes_filtered.csv"
pos_file = "/home/bran/Workspace/misc/hypernet_test4/discretized_data/test_pos.csv"
output_name = "/home/bran/Workspace/misc/hypernet_test4/graph/test_graph.svg"
"""

#simple_display(edge_file, node_file, pos_file, output_name)
