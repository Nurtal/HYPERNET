## importation
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


def run_boruta(file_list_1, file_list_2, output_folder):
    """
    """

    ## parameters
    fname_to_label = {}
    depth = 3
    iteration = 150

    ## create fname to label
    for fname in file_list_1:
        fname = fname.split("/")
        fname = fname[-1]
        fname = output_folder+"/discretized_data/"+fname
        fname = fname.replace(".csv", "_normalized_discretized.csv")
        fname_to_label[fname] = 1
    for fname in file_list_2:
        fname = fname.split("/")
        fname = fname[-1]
        fname = output_folder+"/discretized_data/"+fname
        fname = fname.replace(".csv", "_normalized_discretized.csv")
        fname_to_label[fname] = 2

    ## create dataframe
    matrix = []
    header = []
    for tf in list(fname_to_label.keys()):

        df = pd.read_csv(tf)
        vector = []
        for k in list(df.keys()):
            if(k not in ['centroid_X', 'centroid_Y']):
                if(k not in header):
                    header.append(k)
                vector.append(np.mean(list(df[k])))
        vector.append(fname_to_label[tf])
        matrix.append(vector)

    header.append("LABEL")
    df = pd.DataFrame(matrix, columns=header)


    ## run Boruta
    ## extract features
    features = [f for f in df.columns if f not in ['ID','LABEL']]

    ## prepare dataset
    X = df[features].values
    Y = df['LABEL'].values.ravel()

    ## setup the RandomForrestClassifier as the estimator to use for Boruta
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=depth)

    ## prepare Boruta
    boruta_feature_selector = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=2,
        random_state=4242,
        max_iter = iteration,
        perc = 90
    )

    ## run boruta
    boruta_feature_selector.fit(X, Y)

    # extract selected features
    X_filtered = boruta_feature_selector.transform(X)

    ## save extracted feature
    final_features = list()
    indexes = np.where(boruta_feature_selector.support_ == True)
    for x in np.nditer(indexes):
        final_features.append(features[x])

    ## craft feature output file
    output_filename = output_folder+"/boruta_selected_features.csv"
    output_dataset = open(output_filename, "w")
    output_dataset.write("FEATURE\n")
    for final_f in final_features:
        output_dataset.write(str(final_f)+"\n")
    output_dataset.close()

    ## reduce datasets
    for tf in list(fname_to_label.keys()):
        df = pd.read_csv(tf)
        variable_to_keep = ['centroid_X', 'centroid_Y']
        for var in final_features:
            variable_to_keep.append(var)
        df = df[variable_to_keep]
        df.to_csv(tf, index=False)

    ## return selected features
    return final_features
