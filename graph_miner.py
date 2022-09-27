## importation
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import plotly.express as px
import pprint


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



def get_patterns_to_global_support(file_list, variable_to_keep, min_support, max_len, output_file_name):
    """
    """

    ## craft nodes and edges
    edge_file_list = []
    node_file_list = []
    for target_file in file_list:

        #-> craft edge file list
        edge_file = target_file.replace("discretized_data/", "graph/edges/")
        if(os.path.isfile(edge_file)):
            edge_file_list.append(edge_file)

        #-> craft node file list
        node_file = target_file.replace("discretized_data/", "graph/nodes/")
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
    get_patterns_to_global_support(file_list_1, variable_to_keep, min_support, max_len, extracted_file_name_1)

    ## get pattern for flobal support for file list 2
    get_patterns_to_global_support(file_list_2, variable_to_keep, min_support, max_len, extracted_file_name_2)

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



#hunt_patterns("/home/bran/Workspace/misc/hypernet_test4/graph/edges/test_edges.csv", "/home/bran/Workspace/misc/hypernet_test4/graph/nodes/test_nodes.csv", [["1","2"]])
