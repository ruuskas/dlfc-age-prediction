"""Make features for the machine learning models."""


import numpy as np
import pandas as pd
import networkx as nx

from config import analysis_root, fbands
from filenames import FileNames


# Init filenames
fn = FileNames(analysis_root)

# Load the list of included subjects
participants = pd.read_csv(fn.included_subjects, header=0, index_col=0)
subjects = participants["subject"].to_list()

target_density = 0.2
# Load the AEC and WPLI data for all subjects
for sensor_space in [True, False]:
    for metric in ["wpli", "aec"]:
        graph_features = pd.DataFrame(index=subjects)
        fc_features = pd.DataFrame(index=subjects)
        for fband in fbands.keys():
            fc_data = np.load(fn.fc_data(metric, "tril",
                                         minmax=False,
                                         sensor_space=sensor_space,
                                         fband=fband))
            # fc_data has shape (n_subjects, n_nodes, n_nodes)
            n_nodes = fc_data.shape[1]
            tril_indices = np.tril_indices(n_nodes, k=-1)
            fc_flat = fc_data[:, tril_indices[0], tril_indices[1]]
            np.save(fn.fc_flat(metric, fband, sensor_space), fc_flat)
            fc_features_fband = pd.DataFrame(
                data=fc_flat, index=subjects,
                columns=[f"{metric}_fc_{fband}_{i}"
                         for i in range(fc_flat.shape[1])])
            fc_features = pd.concat([fc_features, fc_features_fband],
                                    axis=1)

            # make binary graphs by keeping only the top n_keep connections
            n_keep = int(target_density *
                         (fc_data.shape[1] * (fc_data.shape[1] - 1)) / 2)
            global_efficiencies = []
            avg_shortest_path_lengths = []
            transitivity_values = []

            for fc_s in fc_data:
                fc_s_flat = fc_s.flatten()
                indices_sorted = np.argsort(fc_s_flat)
                fc_s_flat[indices_sorted[:-n_keep]] = 0
                fc_s_flat[indices_sorted[-n_keep:]] = 1
                fc_s_thresholded = fc_s_flat.reshape(fc_s.shape).astype(int)
                graph = nx.from_numpy_array(fc_s_thresholded)
                global_efficiencies.append(nx.global_efficiency(graph))
                if nx.is_connected(graph):
                    avg_shortest_path_lengths.append(
                        nx.average_shortest_path_length(graph))
                else:
                    largest_component = max(nx.connected_components(graph),
                                            key=len)
                    subgraph = graph.subgraph(largest_component)
                    avg_shortest_path_lengths.append(
                        nx.average_shortest_path_length(subgraph))
                transitivity_values.append(nx.transitivity(graph))

            graph_features[f"{metric}_ge_{fband}"] = global_efficiencies
            graph_features[f"{metric}_avgspl_{fband}"] = avg_shortest_path_lengths
            graph_features[f"{metric}_transitivity_{fband}"] = transitivity_values
        graph_features.to_csv(fn.graph_features(metric, sensor_space))
        fc_features.to_csv(fn.con_features(metric, sensor_space))
