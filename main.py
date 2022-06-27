## imporation
import pandas as pd
import os
import dataset_manager
import graph_manager
import graph_miner


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
        "neighbour_radius":10
    }

    ## load conf file if exist
    if(os.path.isfile(action)):
        df = pd.read_csv(action)
        for index, row in df.iterrows():
            if(row[list(row.keys())[0]] in list(conf_to_val.keys())):
                conf_to_val[row[list(row.keys())[0]]] = row[list(row.keys())[1]]

    ## process variable_to_keep
    if(conf_to_val["variable_to_keep"] != "ALL"):
        conf_to_val["variable_to_keep"] = conf_to_val["variable_to_keep"].split(";")

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




def run(manifest_file, output_folder, action):
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
    min_len = conf_to_val["max_len"]
    variable_to_keep = conf_to_val["variable_to_keep"]
    neighbour_radius = conf_to_val["neighbour_radius"]
    if(variable_to_keep == "ALL"):
        variable_to_keep = extract_all_variables(file_list_1[0])

    ## craft output folder if not exist
    if(not os.path.isdir(output_folder)):
        os.mkdir(output_folder)

    ## prepare dataset
    dataset_manager.load_raw_dataset(label_to_file, output_folder)
    dataset_manager.normalize_dataset(output_folder)
    dataset_manager.simple_discretization(output_folder)

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
        tf = tf.replace(".csv", "_normalized_discretized.csv")

        if(os.path.isfile(tf)):

            #-> update target file list
            file_list_1_discretized.append(tf)

            #-> craft edges
            graph_manager.craft_edge_dataframe(tf, neighbour_radius, output_folder)

            #-> craft nodes
            graph_manager.craft_node_dataframe(tf, output_folder)

    ## craft file list 2
    file_list_2_discretized = []
    for data_file in file_list_2:
        tf = data_file.split("/")
        tf = tf[-1]
        tf = output_folder+"/discretized_data/"+tf
        tf = tf.replace(".csv", "_normalized_discretized.csv")

        if(os.path.isfile(tf)):

            #-> update target file list
            file_list_2_discretized.append(tf)

            #-> craft edges
            graph_manager.craft_edge_dataframe(tf, neighbour_radius, output_folder)

            #-> craft nodes
            graph_manager.craft_node_dataframe(tf, output_folder)

    ## run graph analysis
    graph_miner.run_single_cell_analysis(
        file_list_1_discretized,
        file_list_2_discretized,
        variable_to_keep,
        min_support,
        output_folder
    )




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
    text="HYPERNET - hYperion NETwork"
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
