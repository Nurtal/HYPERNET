## importation
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import plotly.express as px
import pprint
import numpy as np
import re


def extract_connected_nodes(edges_file):
    """
    """

    ## parameters
    subgraph_to_node = {}
    subgraph_cmpt = 0

    ## load data
    df = pd.read_csv(edges_file)

    ## loop over edges
    for index, row in df.iterrows():

        #-> extract information
        source = row["source"]
        target = row["target"]

        #-> check the other subgraph
        extend_existing_subgraph = False
        for subgraph in subgraph_to_node.keys():
            node_list = subgraph_to_node[subgraph]
            if(source in node_list and target not in node_list):
                subgraph_to_node[subgraph].append(target)
                extend_existing_subgraph = True
            if(target in node_list and source not in node_list):
                subgraph_to_node[subgraph].append(source)
                extend_existing_subgraph = True

        #-> create new subgraph
        if(not extend_existing_subgraph):
            subgraph_cmpt+=1
            subgraph = "subgraph_"+str(subgraph_cmpt)
            subgraph_to_node[subgraph] = [source,target]

    ## return subgraph_to_node
    return subgraph_to_node




def convert_subgraph_to_pattern_list(subgraph_to_node, nodes_file, variables_to_keep):
    """
    """

    ## parameters
    node_to_pattern = {}
    pattern_list = []

    ## load nodes file
    df = pd.read_csv(nodes_file)
    if(variables_to_keep != "ALL"):
        df = df[variables_to_keep]

    ## loop over node
    # -> test if df is a dataframe (i.e if multiple variable to describe node)
    # if not, assume it is a serie and then the pattern is composed of only one element
    node_cmpt = 0
    if(isinstance(df, pd.DataFrame)):
        for index, row in df.iterrows():

            #-> identify node
            node_cmpt +=1
            node_name = "cell_"+str(node_cmpt)

            #-> craft pattern
            pattern = ""
            for k in list(row.keys()):
                scalar = row[k]
                pattern+=str(int(scalar))+"-"
            pattern = pattern[:-1]

            #-> update node to features
            node_to_pattern[node_name] = pattern
    else:
        for elt in df:

            #-> identify node
            node_cmpt +=1
            node_name = "cell_"+str(node_cmpt)

            #-> craft pattern
            pattern = elt

            #-> update node to features
            node_to_pattern[node_name] = pattern


    ## loop over subgraph
    for subgraph in subgraph_to_node:
        node_list = subgraph_to_node[subgraph]
        pattern = []
        for node in node_list:
            pattern.append(node_to_pattern[node])
        pattern_list.append(pattern)

    #-> replace node with pattern
    return pattern_list


def fptree_mining(pattern_list, min_support, max_len):
    """
    """

    ## init transaction database
    te = TransactionEncoder()
    te_ary = te.fit(pattern_list).transform(pattern_list)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    ## mine
    frequent_pattern = fpgrowth(df, min_support=min_support, use_colnames=True, max_len=int(max_len))

    ## return frequent_pattern
    return frequent_pattern



def extract_frequent_pattern_from_list(edges_file_list, nodes_file_list, variables_to_keep, min_support, max_len):
    """
    """

    ## parameters
    all_patterns = []

    cmpt = 0
    for edge_file in edges_file_list:
        node_file = nodes_file_list[cmpt]

        ## extract subgraph
        graph_to_node = extract_connected_nodes(edge_file)

        ## extract pattern
        pattern_list = convert_subgraph_to_pattern_list(graph_to_node, node_file, variables_to_keep)

        ## update
        all_patterns += pattern_list
        cmpt+=1

    ## mine frequent pattern
    frequent_pattern = fptree_mining(all_patterns, min_support, max_len)

    ## return frequent pattern
    return frequent_pattern



def get_patterns_to_global_support(file_list, variable_to_keep, min_support, max_len, output_file_name, output_folder):
    """
    """

    ## craft nodes and edges
    edge_file_list = []
    node_file_list = []
    for target_file in file_list:

        #-> look for files
        if(re.search("discretized_data/", target_file)):
            edge_file = target_file.replace("discretized_data/", "graph/edges/")
            node_file = target_file.replace("discretized_data/", "graph/nodes/")
        elif(re.search("raw_data/", target_file)):
            edge_file = target_file.replace("raw_data/", "graph/edges/")
            node_file = target_file.replace("raw_data/", "graph/nodes/")
        else:
            target_file = target_file.split("/")
            target_file = target_file[-1]
            edge_file = output_folder+"/graph/edges/"+str(target_file)
            node_file = output_folder+"/graph/nodes/"+str(target_file)

        #-> craft edge file list
        if(os.path.isfile(edge_file)):
            edge_file_list.append(edge_file)

        #-> craft node file list
        if(os.path.isfile(node_file)):
            node_file_list.append(node_file)

    #-> extract frequent pattern
    extracted_patterns = extract_frequent_pattern_from_list(
            edge_file_list,
            node_file_list,
            variable_to_keep,
            min_support,
            max_len
        )


    ## save dataset
    extracted_patterns.to_csv(output_file_name, index=False)





def compare_support_distribution(data_file1, data_file2, output_file_name):
    """
    """

    ## parameters
    pattern_to_supp1 = {}
    pattern_to_supp2 = {}
    pattern_to_ecart = {}
    sep = ";"

    ## load data file 1
    df1 = pd.read_csv(data_file1)
    for index, row in df1.iterrows():
        p1 = row["itemsets"]
        if(p1 not in pattern_to_supp1.keys()):
            pattern_to_supp1[p1] = row["support"]

    ## load data file 2
    df2 = pd.read_csv(data_file2)
    for index, row in df2.iterrows():
        p2 = row["itemsets"]
        if(p1 not in pattern_to_supp2.keys()):
            pattern_to_supp2[p2] = row["support"]

    ## extend
    for p1 in pattern_to_supp1.keys():
        if(p1 not in pattern_to_supp2.keys()):
            pattern_to_supp2[p1] = 0
    for p2 in pattern_to_supp2.keys():
        if(p2 not in pattern_to_supp1.keys()):
            pattern_to_supp1[p2] = 0

    ## compute ecart
    for p in pattern_to_supp1.keys():
        v1 = pattern_to_supp1[p]
        v2 = pattern_to_supp2[p]
        ecart = abs(v1-v2)
        pattern_to_ecart[p] = ecart

    ## save dataset
    output_file = open(output_file_name, "w")
    header = "itemsets"+str(sep)+"support1"+str(sep)+"support2"+str(sep)+"difference\n"
    output_file.write(header+"\n")
    for p in pattern_to_ecart.keys():
        line = str(p)+str(sep)+str(pattern_to_supp1[p])+str(sep)+str(pattern_to_supp2[p])+str(sep)+str(pattern_to_ecart[p])+"\n"
        output_file.write(line)
    output_file.close()



def compare_multiple_support_distribution(cluster_to_pattern_file, output_file_name):
    """
    """

    ## parameters
    cluster_to_pattern_to_support = {}
    all_patterns = []
    sep = ";"

    ## loop over cluster
    for cluster in cluster_to_pattern_file.keys():
        pattern_file = cluster_to_pattern_file[cluster]
        pattern_to_support = {}

        #-> load patterns
        df = pd.read_csv(pattern_file)
        for index, row in df.iterrows():
            p = row["itemsets"]
            if(p not in pattern_to_support.keys()):
                pattern_to_support[p] = row["support"]
            if(p not in all_patterns):
                all_patterns.append(p)

        #-> update data structure
        cluster_to_pattern_to_support[cluster] = pattern_to_support

    ## extend
    for cluster in cluster_to_pattern_file.keys():
        pattern_to_support = cluster_to_pattern_to_support[cluster]
        for p in all_patterns:
            if(p not in pattern_to_support.keys()):
                pattern_to_support[p] = 0
        cluster_to_pattern_to_support[cluster] = pattern_to_support

    ## save dataset
    output_file = open(output_file_name, "w")
    header = "itemsets"+str(sep)
    for cluster in cluster_to_pattern_to_support.keys():
        header+="support_"+str(cluster)+str(sep)
    header = header[:-1]
    output_file.write(header+"\n")
    for p in all_patterns:
        line = str(p)+str(sep)
        for cluster in cluster_to_pattern_file.keys():
            line += str(cluster_to_pattern_to_support[cluster][p])+str(sep)
        line = line[:-1]
        output_file.write(line+"\n")
    output_file.close()



def hunt_specific_pattern(comparison_file):
    """
    """

    ## parameters
    best_diff_1 = 0
    best_diff_2 = 0
    best_node_1 = "NA"
    best_node_2 = "NA"
    best_supp_1 = 0
    best_supp_2 = 0

    ## load comparison_file
    df = pd.read_csv(comparison_file, sep=";")
    for index, row in df.iterrows():

        #-> extract info
        node = row["itemsets"]
        diff = row["difference"]
        supp1 = row["support1"]
        supp2 = row["support2"]

        ## Hunt
        if(supp1 > supp2):
            if(diff > best_diff_1):
                best_node_1 = node
                best_diff_1 = diff
            if(diff == best_diff_1):
                if(supp1 > best_supp_1):
                    best_node_1 = node
                    best_supp_1 = supp1
        elif(supp2 > supp1):
            if(diff > best_diff_2):
                best_node_2 = node
                best_diff_2 = diff
            if(diff == best_diff_2):
                if(supp2 > best_supp_2):
                    best_node_2 = node
                    best_supp_2 = supp2

    ## return identified itemsets
    return(best_node_1, best_node_2)






def plot_node_as_radar(node, variables, output_file_name):
    """
    """

    ## process node
    node_array = str(node)
    node_array = node_array.replace("frozenset", "")
    node_array = node_array.replace("\"", "")
    node_array = node_array.replace("'", "")
    node_array = node_array.replace("[", "")
    node_array = node_array.replace("]", "")
    node_array = node_array.replace("(", "")
    node_array = node_array.replace(")", "")
    node_array = node_array.replace("{", "")
    node_array = node_array.replace("}", "")
    node_array = node_array.replace(" ", "")
    node_array = node_array.split("-")

    ## craft data structure
    df = pd.DataFrame(dict(r=node_array,theta=variables))

    ## plot figure
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.write_image(output_file_name)





def plot_support_distribution(pattern_support_file):
    """
    """

    ## parameters
    support_to_count = {}

    ## load data file
    df = pd.read_csv(pattern_support_file)

    ## init support to count
    for x in range(1,10):
        k = "0."+str(x)
        support_to_count[k] = 0

    ## craft data
    for index, row in df.iterrows():
        supp = str(row['support'])
        supp = supp[0:3]
        support_to_count[supp] +=1

    ## plot support
    plt.bar(support_to_count.keys(), support_to_count.values())
    plt.title("Support Distribution")
    plt.savefig(pattern_support_file.replace(".csv", ".png"))
    plt.close()



def run_cell_analysis(file_list_1, file_list_2,variable_to_keep, min_support, output_folder, max_len):
    """
    Actually, try to make this not single cell (play with max_len)
    """

    ## parameters
    #max_len = 1
    extracted_file_name_1 = output_folder+"/pattern_ewtracted_1.csv"
    extracted_file_name_2 = output_folder+"/pattern_ewtracted_2.csv"
    comparison_file_name = output_folder+"/pattern_compared.csv"
    node_plot_1 = output_folder+"/node1.png"
    node_plot_2 = output_folder+"/node2.png"

    ## get pattern for flobal support for file list 1
    get_patterns_to_global_support(file_list_1, variable_to_keep, min_support, max_len, extracted_file_name_1, output_folder)

    ## get pattern for flobal support for file list 2
    get_patterns_to_global_support(file_list_2, variable_to_keep, min_support, max_len, extracted_file_name_2, output_folder)

    ## compare support distribution
    compare_support_distribution(extracted_file_name_1, extracted_file_name_2, comparison_file_name)

    ## find most characteristic node for category 1 and 2
    spec_nodes = hunt_specific_pattern(comparison_file_name)
    spec_node_1 = spec_nodes[0]
    spec_node_2 = spec_nodes[1]

    ## plot spec1
    plot_node_as_radar(spec_node_1, variable_to_keep, node_plot_1)

    ## plot spec2
    plot_node_as_radar(spec_node_2, variable_to_keep, node_plot_2)

    ## plot pattern distribution for category 1
    plot_support_distribution(extracted_file_name_1)

    ## plot pattern distribution for category 2
    plot_support_distribution(extracted_file_name_2)



def run_pattern_analysis(cluster_to_file_list, variable_to_keep, min_support, output_folder, max_len):
    """
    TO TEST
    """

    ## paramaters
    cluster_to_pattern_file = {}
    output_file_name = output_folder+"/pattern_comparison.csv"

    ## loop over cluster
    for cluster in cluster_to_file_list.keys():
        file_list = cluster_to_file_list[cluster]
        extracted_file_name = output_folder+"/pattern_extracted_for_"+str(cluster)+".csv"

        # -> extract frequent pattern
        get_patterns_to_global_support(
            file_list,
            variable_to_keep,
            min_support,
            max_len,
            extracted_file_name,
            output_folder
        )

        # -> update dict of pattern files
        cluster_to_pattern_file[cluster] = extracted_file_name

    ## Write comparison table
    compare_multiple_support_distribution(cluster_to_pattern_file, output_file_name)


def generate_radar_profile(cluster_to_file_list, output_folder):
    """
    """

    ## parameters

    ## craft images folder if not exist
    if(not os.path.isdir(output_folder+"/images")):
        os.mkdir(output_folder+"/images")

    ## loop over clusters
    for cluster in cluster_to_file_list.keys():

        ## extract file list
        file_list = cluster_to_file_list[cluster]

        ## init data
        c_to_prop_list = {}

        ## compute proportion of cluster (from cell, not patient) in the file
        for f in file_list:

            ## count cluster
            cmpt = 0
            c_to_count = {}
            df = pd.read_csv(f)
            if("pgraph" in df.keys()):
                df = df.rename(columns={"pgraph":"cluster"})
            for c in list(df['cluster']):
                if(c not in c_to_count.keys()):
                    c_to_count[c] = 1
                else:
                    c_to_count[c] +=1
                cmpt +=1

            ## convert to proportion
            c_to_prop = {}
            for c in c_to_count.keys():
                c_to_prop[c] = float(c_to_count[c]) / cmpt

            ## update big data
            for c in c_to_prop.keys():
                if(c not in c_to_prop_list.keys()):
                    c_to_prop_list[c] = [c_to_prop[c]]
                else:
                    c_to_prop_list[c].append(c_to_prop[c])

        ## compute mean & std
        c_to_mean = {}
        c_to_std_pos = {}
        c_to_std_neg = {}
        for c in c_to_prop_list.keys():
            prop_list = c_to_prop_list[c]
            c_to_mean[c] = np.mean(prop_list)
            c_to_std_pos[c] = np.mean(prop_list) + np.std(prop_list)
            c_to_std_neg[c] = np.mean(prop_list) - np.std(prop_list) 

        ## generate radar plot
        labels = list(c_to_mean.keys())
        values = list(c_to_mean.values())
        values_std_pos = list(c_to_std_pos.values())
        values_std_neg = list(c_to_std_neg.values())

        # Number of variables we're plotting.
        num_vars = len(labels)

        # Split the circle into even parts and save the angles
        # so we know where to put each axis.
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        # The plot is a circle, so we need to "complete the loop"
        # and append the start value to the end.
        values += values[:1]
        values_std_pos += values_std_pos[:1]
        values_std_neg += values_std_neg[:1]
        angles += angles[:1]
        # ax = plt.subplot(polar=True)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        # Draw the outline of our data.
        ax.plot(angles, values, color='red', linewidth=1, label="MEAN")
        ax.plot(angles, values_std_pos, color='red', linewidth=1, label="MEAN + STD", linestyle='dashed')
        ax.plot(angles, values_std_neg, color='red', linewidth=1, label="MEAN - STD", linestyle='dashed')
        
        # Fill it in.
        ax.fill(angles, values, color='red', alpha=0.25)
        # ax.fill(angles, values_std_pos, color='blue', alpha=0.25)
        # ax.fill(angles, values_std_neg, color='red', alpha=0.25)

        # Fix axis to go in the right order and start at 12 o'clock.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        # Draw axis lines for each angle and label. Dirty patch, seems to work
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        plt.title("Mean cell cluster distribution in class "+str(cluster))

        # Add a legend -> Patrice said no need, uncomment the line if needed
        # ax.legend(loc='lower right')

        # save file
        save_file_name = output_folder+"/images/radar_distribution_class"+str(cluster)+".svg"
        plt.savefig(save_file_name)
        plt.close()






def hunt_patterns(edges_file, nodes_file, pattern_list):
    """
    """

    ## parameters
    subgraph_to_node = {}
    node_to_label = {}
    node_save_list = []
    subgraph_cmpt = 0

    ## load data
    df = pd.read_csv(edges_file)

    ## loop over edges
    for index, row in df.iterrows():

        #-> extract information
        source = row["source"]
        target = row["target"]

        #-> check the other subgraph
        extend_existing_subgraph = False
        for subgraph in subgraph_to_node.keys():
            node_list = subgraph_to_node[subgraph]
            if(source in node_list and target not in node_list):
                subgraph_to_node[subgraph].append(target)
                extend_existing_subgraph = True
            if(target in node_list and source not in node_list):
                subgraph_to_node[subgraph].append(source)
                extend_existing_subgraph = True

        #-> create new subgraph
        if(not extend_existing_subgraph):
            subgraph_cmpt+=1
            subgraph = "subgraph_"+str(subgraph_cmpt)
            subgraph_to_node[subgraph] = [source,target]

    ## get node to label
    df_node = pd.read_csv(nodes_file)
    for index, row in df_node.iterrows():

        node_name = "cell_"+str(index+1)
        label_name = row.keys()[0]
        label = row[label_name]
        node_to_label[node_name] = str(label)

    ## loop over subgraph_to_node
    for subgraph in subgraph_to_node.keys():

        #-> extract node list
        node_list = subgraph_to_node[subgraph]

        #-> check patterns to hunt
        for pattern in pattern_list:

            #-> treat only if subgraph can contain pattern
            if(len(node_list) >=len(pattern)):

                #-> check if pattern is contained in subraph
                node_list_labeled = []
                for node in node_list:
                    node_label = node_to_label[node]
                    node_list_labeled.append(node_label)
                node_list_labeled = set(node_list_labeled)
                pattern = set(pattern)
                if set(pattern).issubset(node_list_labeled):
                    for node in node_list:
                        if(node not in node_save_list and node_to_label[node] in pattern):
                            node_save_list.append(node)

    ## create new node file
    node_file_name = nodes_file.replace(".csv", "_filtered.csv")
    node_data = open(node_file_name, "w")
    node_data.write("ID,"+str(label_name)+"\n")
    for node in node_save_list:
        node_data.write(str(node)+","+str(node_to_label[node])+"\n")
    node_data.close()

    ## create edge file
    edge_file_name_out = edges_file.replace(".csv", "_filtered.csv")
    input_edge = open(edges_file, "r")
    output_edge = open(edge_file_name_out, "w")
    output_edge.write("source,target\n")
    for line in input_edge:
        line = line.rstrip()
        line_array = line.split(",")
        if(len(line_array) > 1):
            source = line_array[0]
            target = line_array[1]
            if((source in node_save_list and target in node_save_list)):
                output_edge.write(str(source)+","+str(target)+"\n")
    output_edge.close()
    input_edge.close()



## Summer School demo
"""
pattern_list = [[11.0, 9.0, 2.0, 10.0]]
edge_file = "/home/bran/Workspace/SALIVARY_GLANDS/hypernet_sjs/graph/edges/slide10roi10_mean_phenograph_cluster.csv"
node_file = "/home/bran/Workspace/SALIVARY_GLANDS/hypernet_sjs/graph/nodes/slide10roi10_mean_phenograph_cluster.csv"
hunt_patterns(edge_file, node_file, pattern_list)
"""

#hunt_patterns("/home/bran/Workspace/misc/hypernet_test4/graph/edges/test_edges.csv", "/home/bran/Workspace/misc/hypernet_test4/graph/nodes/test_nodes.csv", [["1","2"]])
