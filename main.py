## imporation
import pandas as pd
import os
import shutil
import dataset_manager
import graph_manager
import graph_miner
import feature_selection
import cell_clustering
import graph_display
import neighboor_analysis

def extract_file_list(manifest_file):
    """
    """

    ## parameters
    label_to_file = {}

    ## load manifest
    df = pd.read_csv(manifest_file)

    ## loop over data
    for index, row in df.iterrows():

        #-> extract info
        fname = row[list(row.keys())[0]]
        label = row[list(row.keys())[1]]

        #-> check if file exist & update dict
        if(os.path.isfile(fname)):
            if(label in label_to_file.keys()):
                label_to_file[label].append(fname)
            else:
                label_to_file[label] = [fname]
        else:
            print("[!] "+str(fname)+" not found")

    ## return category to file
    return label_to_file



def extract_configuration(action):
    """
    """

    ## paramaters
    conf_to_val = {
        "min_support":0.6,
        "max_len":1,
        "variable_to_keep":"ALL",
        "neighbour_radius":10,
        "annotation":"PHENOGRAPH",
        "normalize":True,
        "discretize":True,
        "neighbour_matrix":False
    }

    ## load conf file if exist
    if(os.path.isfile(action)):
        df = pd.read_csv(action)
        for index, row in df.iterrows():
            if(row[list(row.keys())[0]] in list(conf_to_val.keys())):
                conf_to_val[row[list(row.keys())[0]]] = row[list(row.keys())[1]]

    ## process variable_to_keep
    if(conf_to_val["variable_to_keep"] not in ["ALL", "BORUTA"]):
        conf_to_val["variable_to_keep"] = conf_to_val["variable_to_keep"].split(";")

    ## display configuration
    print("[+][CONFIGURATION] - min support =>\t"+str(conf_to_val["min_support"]))
    print("[+][CONFIGURATION] - max lenght =>\t"+str(conf_to_val["max_len"]))
    print("[+][CONFIGURATION] - variable selection =>\t"+str(conf_to_val["variable_to_keep"]))
    print("[+][CONFIGURATION] - neighbour radius =>\t"+str(conf_to_val["neighbour_radius"]))
    print("[+][CONFIGURATION] - annotation strategy =>\t"+str(conf_to_val["annotation"]))
    print("[+][CONFIGURATION] - normalize =>\t"+str(conf_to_val["normalize"]))
    print("[+][CONFIGURATION] - discretize =>\t"+str(conf_to_val["discretize"]))
    print("[+][CONFIGURATION] - neighbour_matrix =>\t"+str(conf_to_val["neighbour_matrix"]))

    ## return configuration
    return conf_to_val



def extract_all_variables(data_file):
    """
    """

    ## parameters
    all_variables = []

    ## load data file
    df = pd.read_csv(data_file)
    for k in list(df.keys()):
        if(k not in ["centroid_X","centroid_Y"] and k not in all_variables):
            all_variables.append(k)

    ## return extracted variables
    return all_variables




def run_old(manifest_file, output_folder, action):
    """
    IN PROGRESS

    -> Work only for 2 labels for now
    """


    ## extract files lists from manifest
    ## extract only 2 first class
    label_to_file = extract_file_list(manifest_file)
    file_list_1 = label_to_file[list(label_to_file.keys())[0]]
    file_list_2 = label_to_file[list(label_to_file.keys())[1]]

    ## extract configuration
    conf_to_val = extract_configuration(action)
    min_support = conf_to_val["min_support"]
    max_len = int(conf_to_val["max_len"])
    variable_to_keep = conf_to_val["variable_to_keep"]
    neighbour_radius = int(conf_to_val["neighbour_radius"])
    annotation = conf_to_val["annotation"]
    if(variable_to_keep == "ALL"):
        variable_to_keep = extract_all_variables(file_list_1[0])

    ## craft output folder if not exist
    if(not os.path.isdir(output_folder)):
        os.mkdir(output_folder)

    ## prepare dataset
    dataset_manager.load_raw_dataset(label_to_file, output_folder)
    dataset_manager.normalize_dataset(output_folder)
    dataset_manager.simple_discretization(output_folder)

    ## perform feature reduction if needed
    if(variable_to_keep == "BORUTA"):
        variable_to_keep = feature_selection.run_boruta(file_list_1, file_list_2, output_folder)

    ## perform phenograph clustering if needed
    if(annotation == "PHENOGRAPH"):
        file_list_1.extend(file_list_2)
        file_list = file_list_1
        cell_clustering.annotation_with_pehnograph(file_list, variable_to_keep, output_folder)
        variable_to_keep = "cluster"

    ## init graph folder
    if(not os.path.isdir(output_folder+"/graph")):
        os.mkdir(output_folder+"/graph")
    if(not os.path.isdir(output_folder+"/graph/nodes")):
        os.mkdir(output_folder+"/graph/nodes")
    if(not os.path.isdir(output_folder+"/graph/edges")):
        os.mkdir(output_folder+"/graph/edges")

    ## craft graph
    ## craft file list 1
    file_list_1_discretized = []
    for data_file in file_list_1:
        tf = data_file.split("/")
        tf = tf[-1]
        tf = output_folder+"/discretized_data/"+tf
        if(annotation == "PHENOGRAPH"):
            tf = tf.replace(".csv", "_phenograph_cluster.csv")
        else:
            tf = tf.replace(".csv", "_normalized_discretized.csv")

        if(os.path.isfile(tf)):

            #-> update target file list
            file_list_1_discretized.append(tf)

            #-> craft edges
            graph_manager.craft_edge_dataframe(tf, neighbour_radius, output_folder)

            #-> craft nodes
            graph_manager.craft_node_dataframe(tf, output_folder)

            #-> generate graphe image
            edge_file = tf.split("/")
            edge_file = edge_file[-1]
            edge_file = output_folder+"/graph/edges/"+edge_file
            node_file = tf.split("/")
            node_file = node_file[-1]
            node_file = output_folder+"/graph/nodes/"+node_file
            pos_file = tf.split("/")
            pos_file = pos_file[-1]
            pos_file = output_folder+"/discretized_data/"+pos_file
            output_name = tf.split("/")
            output_name = output_name[-1]
            output_name = output_folder+"/graph/"+output_name
            output_name = output_name.replace(".csv", "_graph.svg")

            #-> generate graph
            graph_display.simple_display(edge_file, node_file, pos_file, output_name)

    ## craft file list 2
    file_list_2_discretized = []
    for data_file in file_list_2:
        tf = data_file.split("/")
        tf = tf[-1]
        tf = output_folder+"/discretized_data/"+tf
        if(annotation == "PHENOGRAPH"):
            tf = tf.replace(".csv", "_phenograph_cluster.csv")
        else:
            tf = tf.replace(".csv", "_normalized_discretized.csv")

        if(os.path.isfile(tf)):

            #-> update target file list
            file_list_2_discretized.append(tf)

            #-> craft edges
            graph_manager.craft_edge_dataframe(tf, neighbour_radius, output_folder)

            #-> craft nodes
            graph_manager.craft_node_dataframe(tf, output_folder)

            #-> generate graphe image
            edge_file = tf.split("/")
            edge_file = edge_file[-1]
            edge_file = output_folder+"/graph/edges/"+edge_file
            node_file = tf.split("/")
            node_file = node_file[-1]
            node_file = output_folder+"/graph/nodes/"+node_file
            pos_file = tf.split("/")
            pos_file = pos_file[-1]
            pos_file = output_folder+"/discretized_data/"+pos_file
            output_name = tf.split("/")
            output_name = output_name[-1]
            output_name = output_folder+"/graph/"+output_name
            output_name = output_name.replace(".csv", "_graph.svg")

            #-> generate graph
            graph_display.simple_display(edge_file, node_file, pos_file, output_name)

    ## run graph analysis
    graph_miner.run_cell_analysis(
        file_list_1_discretized,
        file_list_2_discretized,
        variable_to_keep,
        min_support,
        output_folder,
        max_len
    )



def run(manifest_file, output_folder, action):
    """
    IN PROGRESS

    -> Try to get this thing working for n cluster
    """

    ## paramaters
    sub_out_folder = "raw_data"

    ## extract files lists from manifest
    label_to_file = extract_file_list(manifest_file)
    all_files = []
    for k in label_to_file.keys():
        flist = label_to_file[k]
        for f in flist:
            if(f not in all_files):
                all_files.append(f)

    ## extract configuration
    conf_to_val = extract_configuration(action)
    min_support = conf_to_val["min_support"]
    max_len = int(conf_to_val["max_len"])
    variable_to_keep = conf_to_val["variable_to_keep"]
    neighbour_radius = int(conf_to_val["neighbour_radius"])
    annotation = conf_to_val["annotation"]
    normalize = conf_to_val["normalize"]
    discretize = conf_to_val["discretize"]
    if(variable_to_keep == "ALL"):
        variable_to_keep = extract_all_variables(all_files[0])

    ## craft output folder if not exist
    if(not os.path.isdir(output_folder)):
        os.mkdir(output_folder)

    ## export config file to output folder (if action is a file)
    if(os.path.isfile(action)):
        destination = output_folder+"/configuration.csv"
        shutil.copy(action, destination)

    ## prepare dataset
    dataset_manager.load_raw_dataset(label_to_file, output_folder)
    if(str(normalize) == "True"):
        dataset_manager.normalize_dataset(output_folder)
        sub_out_folder = "normalized_data"
    if(str(discretize) == "True"):
        dataset_manager.simple_discretization(output_folder)
        sub_out_folder = "discretized_data"

    ## perform feature reduction if needed
    if(variable_to_keep == "BORUTA"):
        variable_to_keep = feature_selection.run_boruta(file_list_1, file_list_2, output_folder)

    ## perform phenograph clustering if needed
    if(annotation == "PHENOGRAPH"):
        file_list_1.extend(file_list_2)
        file_list = file_list_1
        cell_clustering.annotation_with_pehnograph(file_list, variable_to_keep, output_folder)
        variable_to_keep = "cluster"

    ## init graph folder
    if(not os.path.isdir(output_folder+"/graph")):
        os.mkdir(output_folder+"/graph")
    if(not os.path.isdir(output_folder+"/graph/nodes")):
        os.mkdir(output_folder+"/graph/nodes")
    if(not os.path.isdir(output_folder+"/graph/edges")):
        os.mkdir(output_folder+"/graph/edges")

    ## run file analysis
    graph_miner.generate_radar_profile(label_to_file, output_folder)

    ## craft graph
    for label in label_to_file.keys():

        ## get files
        file_list = label_to_file[label]

        ## craft graph
        ## craft file list 1
        #file_list_1_discretized = []
        for data_file in file_list:
            tf = data_file.split("/")
            tf = tf[-1]
            tf = output_folder+"/"+sub_out_folder+"/"+tf

            if(annotation == "PHENOGRAPH"):
                tf = tf.replace(".csv", "_phenograph_cluster.csv")
            elif(normalize == "True" and discretize == "True"):
                tf = tf.replace(".csv", "_normalized_discretized.csv")

            if(os.path.isfile(tf)):

                #-> update target file list
                #file_list_1_discretized.append(tf)

                #-> craft edges
                graph_manager.craft_edge_dataframe(tf, neighbour_radius, output_folder)

                #-> craft nodes
                graph_manager.craft_node_dataframe(tf, output_folder)

                #-> generate graphe image
                edge_file = tf.split("/")
                edge_file = edge_file[-1]
                edge_file = output_folder+"/graph/edges/"+edge_file
                node_file = tf.split("/")
                node_file = node_file[-1]
                node_file = output_folder+"/graph/nodes/"+node_file
                pos_file = tf.split("/")
                pos_file = pos_file[-1]
                pos_file = output_folder+"/"+sub_out_folder+"/"+pos_file
                output_name = tf.split("/")
                output_name = output_name[-1]
                output_name = output_folder+"/graph/"+output_name
                output_name = output_name.replace(".csv", "_graph.svg")

                #-> generate graph
                graph_display.simple_display(edge_file, node_file, pos_file, output_name)


    ## run distance matrix generation
    if(str(conf_to_val["neighbour_matrix"]) == "True"):

        #-> craft images subfolder if not exist
        if(not os.path.isdir(output_folder+"/images")):
            os.mkdir(output_folder+"/images")

        #-> get file list for each label
        for cluster in label_to_file.keys():
            file_list = label_to_file[cluster]
            heatmap_image_file = output_folder+"/images/"+"class_"+str(cluster)+"_neigboor_matrix.png"
            neighboor_analysis.run_on_multiple_files(file_list, heatmap_image_file, float(conf_to_val["neighbour_radius"]))

    ## run graph analysis
    graph_miner.run_pattern_analysis(
        label_to_file,
        variable_to_keep,
        min_support,
        output_folder,
        max_len
    )
    """
    graph_miner.run_cell_analysis(
        file_list_1_discretized,
        file_list_2_discretized,
        variable_to_keep,
        min_support,
        output_folder,
        max_len
    )
    """



##------##
## MAIN ########################################################################
##------##
if __name__=='__main__':

    ## importation
    import sys
    import getopt
    from colorama import init
    init(strip=not sys.stdout.isatty())
    from termcolor import cprint
    from pyfiglet import figlet_format

    ## catch arguments
    argv = sys.argv[1:]

    ## parse arguments
    manifest_file = ''
    output_folder = ''
    action = ''
    try:
       opts, args = getopt.getopt(argv,"hm:o:c:",["mfile=","ofolder=", "conf="])
    except getopt.GetoptError:
       display_help()
       sys.exit(2)
    for opt, arg in opts:
       if opt in ('-h', '--help'):
           display_help()
           sys.exit()
       elif opt in ("-m", "--mfle"):
          manifest_file = arg
       elif opt in ("-o", "--ofolder"):
           output_folder = arg
       elif opt in ("-c", "--conf"):
           action = arg

    ## display cool banner
    text="HYPERNET - HYPERion NETwork"
    cprint(figlet_format(text, font="standard"), "blue")

    ## check that all arguments are present
    if(manifest_file == ''):
        print("[!] No input file detected")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()
    if(output_folder == ''):
        print("[!] No output folder detected")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()
    if(action == ''):
        print("[!] No action detected")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()

    ## perform run
    run(manifest_file, output_folder, action)
