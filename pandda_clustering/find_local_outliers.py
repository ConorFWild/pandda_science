import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import joblib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scipy.optimize import differential_evolution, shgo
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

import hdbscan
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, DBSCAN

import matplotlib.pyplot as plt

from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap

from mdc.dataset import PanDDADataset
from mdc.xmap import PanDDAXMap, sample


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--data_dirs",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-o", "--out_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        parser.add_argument("--mtz_regex",
                            type=str,
                            help="Number of processes to start",
                            default="dimple.mtz",
                            )

        parser.add_argument("--pdb_regex",
                            type=str,
                            help="Number of processes to start",
                            default="dimple.pdb",
                            )
        parser.add_argument("--structure_factors",
                            type=str,
                            help="Number of processes to start",
                            default="FWT,PHWT",
                            )

        args = parser.parse_args()

        self.out_dir = Path(args.out_dir)
        self.pdb_regex = str(args.pdb_regex)
        self.mtz_regex = str(args.mtz_regex)
        self.structure_factors = str(args.structure_factors)
        self.data_dirs = Path(args.data_dirs)


class ClusterFSModel:
    def __init__(self,
                 input_dir,
                 output_dir,
                 ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cluster_table_path = output_dir / "clustering.csv"
        self.cluster_html_path = output_dir / "clustering.html"
        self.initial_model_dirs = {path.name: path for path in input_dir.glob("*")}


def map_dict(f, dictionary):
    # keys = list(dictionary.keys())
    # values = list(dictionary.values())

    results = {}
    for key, value in dictionary.items():
        results[key] = f(value)

    return results


def truncate_dataset():
    pass


def embed(numpy_maps):
    # Convert to sample by feature
    sample_by_feature = np.vstack([xmap.flatten() for xmap in numpy_maps])

    # Reduce dimension by PCA
    pca = PCA(n_components=min(50, len(numpy_maps)))
    sample_by_feature_pca = pca.fit_transform(sample_by_feature)

    # Reduce dimension by TSNE
    tsne = TSNE(n_components=2,
                method="exact",
                )
    sample_by_feature_tsne = tsne.fit_transform(sample_by_feature_pca)

    return sample_by_feature_tsne


def cluster(distances):
    # Perform initial clustering
    distances = np.array(distances).reshape(-1, 1)
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True,
                                prediction_data=True,
                                min_cluster_size=5,
                                )
    # labels = clusterer.fit(xmap_embedding.astype(np.float64)).labels_

    clusterer.fit(np.array(distances).astype(np.float64))

    return clusterer.labels_


def sample_and_measure(reference_sample_map,
                       moving_xmap,
                       parameters,
                       ):
    sample_map = sample(moving_xmap,
                        parameters,
                        )
    distance = np.mean(np.abs(reference_sample_map - sample_map))

    return distance


def align_map(reference_reflections, moving_reflections, reference_centre, moving_centre, delta=3):
    bounds = ((moving_centre[0] - delta, moving_centre[0] + delta),
              (moving_centre[1] - delta, moving_centre[1] + delta),
              (moving_centre[2] - delta, moving_centre[2] + delta),
              (0, 2 * np.pi),
              (0, 2 * np.pi),
              (0, 2 * np.pi),
              )

    reference_xmap = PanDDAXMap.from_reflections(reference_reflections)

    moving_xmap = PanDDAXMap.from_reflections(moving_reflections)

    reference_sample_map = sample(reference_xmap,
                                  (reference_centre[0],
                                   reference_centre[1],
                                   reference_centre[2],
                                   np.pi,
                                   np.pi,
                                   np.pi,
                                   ),
                                  )

    # result = differential_evolution(lambda x: sample_and_measure(reference_sample_map,
    #                                                              moving_xmap,
    #                                                              x,
    #                                                              ),
    #                                 bounds,
    #                                 maxiter=20,
    #                                 )
    result = shgo(lambda x: sample_and_measure(reference_sample_map,
                                               moving_xmap,
                                               x,
                                               ),
                  bounds,
                  )

    del reference_xmap
    del moving_xmap
    del reference_reflections
    del moving_reflections
    del reference_centre
    del moving_centre
    gc.collect()

    print("\t\tResult parameters: {}".format(result.x))
    print("\t\tResult parameters: {}".format(result.fun))

    aligned_map = sample(moving_xmap,
                         result.x,
                         )

    return aligned_map, result.fun


def align_maps(reference_dataset, datasets, alignments):
    results = []

    pool = joblib.Parallel(n_jobs=20, verbose=10)

    tasks = []
    for dtag, alignment in alignments.items():
        print("\t\tAligning map {}".format(dtag))

        reference_reflections = reference_dataset.reflections
        moving_reflections = datasets[dtag].reflections

        tasks.append([reference_reflections,
                           moving_reflections,
                           alignments[reference_dataset.id],
                           alignments[dtag],]
                     )


        # result = align_map(reference_reflections,
        #                    moving_reflections,
        #                    alignments[reference_dataset.id],
        #                    alignments[dtag],
        #                    )

        # results.append(result)

    results = pool(joblib.delayed(align_map)(arg[0],
                                            arg[1],
                                            arg[2],
                                            arg[3],
                                            )
                                  for arg
                                  in tasks
                                  )


        # map_dict(lambda x: align_map(reference_dataset, x),
        #                datasets,
        #                )

    xmaps = [result[0] for result in results]
    distances = [result[1] for result in results]

    return xmaps, distances


def get_reference_dataset(datasets):
    # reference_dataset = min(list(datasets.values()),
    #                         key=lambda x: x.get_resolution_high(),
    #                         )
    reference_dataset = np.random.choice(list(datasets.values()))
    return reference_dataset


def more_that_one_cluster(clusters):
    clusters_non_negative = clusters[clusters >= 0]
    if len(np.unique(clusters_non_negative)) == 0:
        return False
    if len(np.unique(clusters_non_negative)) == 1:
        return False
    if len(np.unique(clusters_non_negative)) > 1:
        return True


def get_alignments(residues):
    alignments = {}
    for dtag, residue in residues.items():
        coord = np.mean(np.array([(atom.pos[0], atom.pos[1], atom.pos[2]) for atom in residue]), axis=0, )
        alignments[dtag] = coord

    return alignments


def get_unclustered_datasets(reference_dataset,
                             truncated_datasets,
                             cluster_distances,
                             ):
    reference_id = reference_dataset.id

    key_array = np.array(list(truncated_datasets.keys()))

    arg = np.argwhere(key_array == reference_id)

    print(cluster_distances)

    print(arg)

    reference_cluster = cluster_distances[arg]
    if reference_cluster == -1:
        print("\t\tReference dataset is on its own!")
        unclustered_datasets = {dtag: truncated_datasets[dtag]
                                for i, dtag
                                in enumerate(truncated_datasets)
                                if dtag != reference_id
                                }

    else:
        unclustered_datasets = {dtag: truncated_datasets[dtag]
                                for i, dtag
                                in enumerate(truncated_datasets)
                                if cluster_distances[i] != reference_cluster
                                }

    return unclustered_datasets, reference_cluster


def cluster_maps_hdbscan(aligned_maps):
    flatmaps = np.vstack([aligned_map.flatten() for aligned_map in aligned_maps])
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True)

    clusterer.fit(flatmaps.astype(np.float64))

    return clusterer.labels_


def plot(aligned_maps,
         cluster_distances,
         path,
         ):
    embeding = embed(aligned_maps)

    fig, ax = plt.subplots()
    ax.scatter(embeding[:, 0],
               embeding[:, 1],
               c=cluster_distances,
               )
    fig.savefig(str(path))
    plt.close(fig)


def cluster_embedded_maps_hdbscan(aligned_maps):
    embeding = embed(aligned_maps)
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True,
                                # min_cluster_size=5,
                                )

    clusterer.fit(embeding.astype(np.float64))

    return clusterer.labels_


def cluster_dbscan(aligned_maps):
    embedding = np.vstack([xmap.flatten() for xmap in aligned_maps])
    clusterer = DBSCAN(eps=15)
    clusterer.fit(embedding.astype(np.float64))

    return clusterer.labels_


def cluster_hdbscan(aligned_maps):
    embedding = np.vstack([xmap.flatten() for xmap in aligned_maps])
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True,
                                # min_cluster_size=5,
                                )

    clusterer.fit(embedding.astype(np.float64))

    return clusterer.labels_


def cluster_embedded_maps_optics(aligned_maps):
    # embeding = embed(aligned_maps)
    embedding = np.vstack([xmap.flatten() for xmap in aligned_maps])
    clusterer = OPTICS()

    clusterer.fit(embedding.astype(np.float64))

    return clusterer.labels_


def cluster_vbgm(aligned_maps):
    # sample_by_features = np.vstack([xmap.flatten() for xmap in aligned_maps])
    embedding = embed(aligned_maps)
    clusterer = BayesianGaussianMixture(n_components=10)
    return clusterer.fit_predict(embedding)


def cluster_cutoff(aligned_maps, distances, cutoff=0.15):
    clusters = []
    for distance in distances:
        if distance < cutoff:
            clusters.append(1)
        else:
            clusters.append(0)

    return np.array(clusters, dtype=np.int)


# def cluster_angles(aligned_maps, reference_idx, cutoff=32*32*32*0.1):
#
#     reference_xmap = aligned_maps[reference_idx]
#
#
#     clusters = []
#     for xmap in aligned_maps:
#         mask_less = np.less(xmap, reference_xmap)
#         mask_equals = np.equal(xmap, reference_xmap)
#         mask_greater = np.greater(xmap, reference_xmap)
#         print(mask_less.shape)
#         print(np.less(xmap, reference_xmap).shape)
#         print(np.sum(np.less(xmap, reference_xmap)).shape)
#
#         diff = sum(mask_less) - sum(mask_greater)
#
#         print(diff)
#
#         if diff > cutoff:
#             clusters.append(1)
#
#         elif diff < -1*cutoff:
#             clusters.append(1)
#
#         else:
#             clusters.append(0)
#
#     return np.array(clusters, dtype=np.int)

def plot_distributions(aligned_maps, out_dir):
    fig, axs = plt.subplots(nrows=10,
                            ncols=10,
                            figsize=(10,
                                     10,
                                     )
                            )

    for i in range(10):
        for j in range(10):
            xyz = (np.random.randint(0, 32), np.random.randint(0, 32), np.random.randint(0, 32))
            dist = [aligned_map[xyz[0], xyz[1], xyz[2]] for aligned_map in aligned_maps]
            axs[i, j].hist(dist)
            # axs[i, j].set_title(str(dtag))
            # if j == 9:
            #     break

    fig.savefig(str(out_dir / "dists.png"))
    plt.close(fig)


def get_extrema(aligned_maps, p=0.05):
    extrema = []
    num_maps = len(aligned_maps)
    ndarray = np.stack(aligned_maps, axis=0)
    sorted_array = np.argsort(ndarray, axis=0)
    for i in range(len(aligned_maps)):
        img = sorted_array[i, :, :, :]
        high = np.count_nonzero(img > num_maps * (1 - p))
        low = np.count_nonzero(img < num_maps * (p))
        outliers = high + low
        print(high, low, outliers)
        if high > 1700:
            extrema.append(1)
        else:
            extrema.append(0)

    return extrema


def cluster_datasets_dep(truncated_datasets,
                         residues,
                         out_dir,
                         ):
    print("\tClustering {} datasets".format(len(truncated_datasets)))

    reference_dataset = get_reference_dataset(truncated_datasets)

    alignments = get_alignments(residues,
                                )

    aligned_maps, distances = align_maps(reference_dataset,
                                         truncated_datasets,
                                         alignments,
                                         )
    print("\tRange of distances is {} {}".format(min(distances),
                                                 max(distances),
                                                 ))

    # cluster_distances = cluster(distances)
    cluster_distances = []
    # for distance in distances:
    #     if distance< 0.1:
    #         cluster_distances.append(0)
    #     else:
    #         cluster_distances.append(1)
    # cluster_distances = np.array(cluster_distances)
    # cluster_distances = cluster_embedded_maps_hdbscan(aligned_maps)
    # cluster_distances = cluster_vbgm(aligned_maps)
    # cluster_distances = cluster_embedded_maps_optics(aligned_maps)
    # cluster_distances = cluster_dbscan(aligned_maps)
    # cluster_distances = cluster_cutoff(aligned_maps,
    #                                    distances,
    #                                    )
    # cluster_distances = cluster_hdbscan(aligned_maps)
    # cluster_distances = cluster_angles(aligned_maps,
    #                                    list(residues.keys()).index(reference_dataset.id),
    #                                    )
    extrema = get_extrema(aligned_maps)

    return [{dtag: aligned_maps[i] for i, dtag in enumerate(list(residues.keys())) if extrema[i] == 0}] + [
        {dtag: aligned_maps[i]} for i, dtag in enumerate(list(residues.keys())) if extrema[i] == 1]

    # plot_distributions(aligned_maps,
    #                    out_dir,
    #                    )
    #
    # print("\tDiscovered {} unique clusters".format(np.unique(cluster_distances,
    #                                                          return_counts=True,
    #                                                          )))

    # unclustered_datasets, reference_cluster = get_unclustered_datasets(reference_dataset,
    #                                                                    truncated_datasets,
    #                                                                    cluster_distances,
    #                                                                    )
    # print("\tGot {} unclustered datasets".format(len(unclustered_datasets)))
    # print("\tReference cluster is {}".format(reference_cluster))
    #
    # cluster_maps = {dtag: aligned_maps[i]
    #                 for i, dtag
    #                 in enumerate(list(residues.keys()))
    #                 if cluster_distances[i] == reference_cluster
    #                 }
    #
    # uncluster_maps = {dtag: aligned_maps[i]
    #                   for i, dtag
    #                   in enumerate(list(residues.keys()))
    #                   if cluster_distances[i] != reference_cluster
    #                   }
    # unclustered_residues = {dtag: residues[dtag] for dtag in unclustered_datasets}
    #
    # plot(aligned_maps,
    #      cluster_distances,
    #      out_dir / "{}.png".format(reference_dataset.id),
    #      )
    #
    # if reference_cluster == -1:
    #     reference_map = aligned_maps[list(residues.keys()).index(reference_dataset.id)]
    #     return [{reference_dataset.id: reference_map}] + [x for x in cluster_datasets(unclustered_datasets,
    #                                                                                   unclustered_residues,
    #                                                                                   out_dir)]
    #
    # # if single is True:
    # #     return [cluster_maps] + [{dtag: xmap} for dtag, xmap in uncluster_maps.items()]
    #
    # if more_that_one_cluster(cluster_distances):
    #     return [cluster_maps] + [x for x in cluster_datasets(unclustered_datasets, unclustered_residues, out_dir)]
    # else:
    #     return [cluster_maps] + [{dtag: xmap} for dtag, xmap in uncluster_maps.items()]
    # #
    # # if len(unclustered_datasets) < 5:
    # #     print("Less than 5 maps: cannot cluster any more!")
    # #     return [cluster_maps] + [{dtag: xmap} for dtag, xmap in uncluster_maps.items()]


def gaussian_distance(sample, means, precs):
    # return np.array([np.linalg.norm(sample) for sample in samples])
    return mahalanobis(sample.flatten(), means, precs)


def probability_distance(samples, model):
    return model.score_samples(samples)


def sample_outlier_distance(model):
    outlier_distances = []
    for i in range(10):
        samples = model.sample(1000)
        print(len(samples))
        print(samples[0].shape)
        # distances = []
        # for i in range(samples[0].shape[0]):
        #     distance = gaussian_distance(samples[0][i, :], model)
        #     print(distance)
        #     distances.append(distance)
        distances = model.score_samples(samples[0])
        print(distances.shape)
        print(distances)
        sorted_distances = np.sort(distances)
        print(sorted_distances.shape)
        print(sorted_distances)
        outlier_distance = np.quantile(sorted_distances, 0.02)
        # print(outlier_distance.shape)
        print(outlier_distance)
        outlier_distances.append(outlier_distance)

    outlier_distance = np.mean(outlier_distances)

    return outlier_distance


def map_list(func, lst):
    results = []
    for x in lst:
        result = func(x)
        print(result)
        results.append(result)
    return results


def map_list_parallel(func, lst):
    return joblib.Parallel(n_jobs=20, verbose=10)(joblib.delayed(func)(x) for x in lst)


def classify_distance(xmap, outlier_distance, means, precs):
    distance = gaussian_distance(xmap, means, precs)
    # distance = probability_distance(xmap.reshape(1,-1), model)
    print(distance)
    if distance < outlier_distance:
        return 0
    else:
        return 1


def cluster_datasets(truncated_datasets,
                     residues,
                     out_dir,
                     ):
    print("\tClustering {} datasets".format(len(truncated_datasets)))

    reference_dataset = get_reference_dataset(truncated_datasets)

    alignments = get_alignments(residues,
                                )

    aligned_maps, distances = align_maps(reference_dataset,
                                         truncated_datasets,
                                         alignments,
                                         )
    print("\tRange of distances is {} {}".format(min(distances),
                                                 max(distances),
                                                 ))

    model = GaussianMixture(n_components=1, covariance_type="diag", verbose=2)
    model.fit(np.vstack([aligned_map.flatten() for aligned_map in aligned_maps]))
    # outlier_distance = sample_outlier_distance(model)
    outlier_distance = np.sqrt(chi2.ppf(0.999, model.means_.size))
    print("Outlier distance: {}".format(outlier_distance))
    precs = np.diag(model.precisions_[0, :])
    means = model.means_[0, :].flatten()
    outliers = map_list(lambda x: classify_distance(x, outlier_distance, means, precs),
                        aligned_maps,
                        )
    # outliers = []
    # for xmap in aligned_maps:
    #     distance = gaussian_distance(xmap, model)
    #     # distance = probability_distance(xmap.reshape(1,-1), model)
    #     print(distance)
    #     if distance < outlier_distance:
    #         outliers.append(1)
    #     else:
    #         outliers.append(0)

    return [{dtag: aligned_maps[i] for i, dtag in enumerate(list(residues.keys())) if outliers[i] == 0}] + [
        {dtag: aligned_maps[i]} for i, dtag in enumerate(list(residues.keys())) if outliers[i] == 1]


def get_comparable_residues(datasets,
                            residue_id,
                            ):
    residue_map = {}

    for dtag, dataset in datasets.items():
        try:
            residue = dataset.structure.structure[residue_id[0]][residue_id[1]][str(residue_id[2])]
        except Exception as e:
            print(e)
            residue = None

        residue_map[dtag] = residue

    return residue_map


def visualise_clusters(clusters,
                       path,
                       ):
    fig, axs = plt.subplots(nrows=min(50, len(clusters)),
                            ncols=10,
                            figsize=(10,
                                     min(len(clusters), 50),
                                     )
                            )

    for i, cluster in enumerate(clusters):
        for j, dtag in enumerate(cluster.keys()):
            image = np.mean(cluster[dtag], axis=0)
            axs[i, j].imshow(image)
            axs[i, j].set_title(str(dtag))
            if j == 9:
                break
        if i == min(49, len(clusters) - 1):
            break

    fig.savefig(str(path))
    plt.close(fig)

    # fig = make_subplots(rows=min(50, len(clusters)),
    #                     cols=10,
    #                     )
    #
    # fig.add_trace(
    #     go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    #     row=1, col=1
    # )
    #
    # fig.add_trace(
    #     go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    #     row=1, col=2
    # )
    #
    # fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")


def try_load(path, mtz_regex="dimple_twin.mtz", pdb_regex="dimple_twin.pdb"):
    try:
        dataset = PanDDADataset.from_dir(path,
                                         mtz_regex=mtz_regex,
                                         pdb_regex=pdb_regex,
                                         )
        return dataset
    except Exception as e:
        print(e)
        return None


def output_csv(clusters,
               path,
               ):
    records = []
    for i, cluster in enumerate(clusters):
        for dtag, xmap in cluster.items():
            record = {}
            record["dtag"] = dtag
            record["cluster"] = i
            records.append(record)

    table = pd.DataFrame(records)
    print(table.head())

    table.to_csv(str(path))

    return table


def make_joint_table(tables):
    for resid, table in tables.items():
        table["model"] = resid[0]
        table["chain"] = resid[1]
        table["num"] = resid[2]

    joint_tables = pd.concat(list(tables.values()))

    return joint_tables


def stackplot(tables,
              path,
              ):
    # One bar per cluster, one x per residue
    core = []
    outliers = []
    resids = []
    for resid, table in tables.items():
        core.append(len(table[table["cluster"] == 0]))
        outliers.append(len(table[table["cluster"] != 0]))
        resids.append(str(resid))

    fig = go.Figure(data=[
        go.Bar(name='core', x=resids, y=core),
        go.Bar(name='outliers', x=resids, y=outliers)
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.write_image(str(path))


def make_outlier_table(joint_table):
    dtags = joint_table["dtag"].unique()

    records = []
    outlier_counts = {}
    for dtag in dtags:
        dtag_table = joint_table[joint_table["dtag"] == dtag]
        num_outliers = len(dtag_table[dtag_table["cluster"] != 0])
        num_residues = len(dtag_table)
        outlier_counts[dtag] = float(num_outliers) / float(num_residues)
        record = {}
        record["dtag"] = dtag
        record["outlier_fraction"] = outlier_counts[dtag]

    return pd.DataFrame(records)


def main():
    args = Args()

    fs = ClusterFSModel(args.data_dirs,
                        args.out_dir,
                        )

    datasets = map_dict(try_load,
                        fs.initial_model_dirs,
                        )
    datasets = {dtag: dataset for dtag, dataset in datasets.items() if dataset is not None}
    # datasets = {item[0]: item[1] for i, item in enumerate(datasets.items()) if i < 100}
    print("\tNumber of datasets is {}".format(len(datasets)))

    datasets_res_high = min([x.get_resolution_high() for x in datasets.values()])
    print("\tHigh resolution is {}".format(datasets_res_high))

    reference_dataset = get_reference_dataset(datasets)
    print("\tReference dataset name is {}".format(reference_dataset.id))

    map_dict(lambda x: x.reflections.truncate_reflections(datasets_res_high),
             datasets,
             )
    truncated_datasets = datasets
    print("\tAfter truncation {} datasets".format(len(truncated_datasets)))

    all_residues = {}
    for model in reference_dataset.structure.structure:
        for chain in model:
            for residue in chain:
                if residue.het_flag != "A":
                    continue
                residue_id = (model.name, chain.name, residue.seqid.num)
                all_residues[residue_id] = residue

    sampled_residues = {list(all_residues.keys())[i]: list(all_residues.values())[i]
                        for i
                        in range(len(all_residues))
                        if i in np.random.choice(range(len(all_residues)),
                                                     50,
                                                     replace=False,
                                                     )
                        }
    print(sampled_residues)

    tables = {}
    for residue_id, residue in sampled_residues.items():
        try:
            print("\tWorking on dataset: {}".format(residue_id))
    
            residues = get_comparable_residues(datasets,
                                               residue_id,
                                               )
            print("\tGot {} residues".format(len(residues)))
            print(residues)
    
            residues = {dtag: res[0] for dtag, res in residues.items() if res is not None}
            print("\tGot {} residues".format(len(residues)))
            print(residues)
    
            clusters = cluster_datasets(truncated_datasets,
                                        residues,
                                        fs.output_dir,
                                        )
            print("\tGot {} clusters".format(len(clusters)))
    
            table = output_csv(clusters,
                               fs.output_dir / "{}_{}_{}.csv".format(residue_id[0],
                                                                     residue_id[1],
                                                                     residue_id[2], ),
                               )
            tables[residue_id] = table
    
            visualise_clusters(clusters,
                               fs.output_dir / "{}_{}_{}.png".format(residue_id[0],
                                                                     residue_id[1],
                                                                     residue_id[2], ),
                               )
        except Exception as e:
            print(e)
    # Make joint table
    joint_table = make_joint_table(tables)
    joint_table.to_csv(str(fs.output_dir / "joint_table.csv"))

    # Make stackplot
    stackplot(tables,
              fs.output_dir / "stackplot.png",
              )

    # Make outlier table
    outlier_table = make_outlier_table(joint_table)
    outlier_table.to_csv(fs.output_dir / "outliers.csv", )


if __name__ == "__main__":
    main()
