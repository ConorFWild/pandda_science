import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution, shgo

import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    sample_by_feature = np.vstack(numpy_maps)

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
    distances = distances.reshape(-1,1)
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True,
                                prediction_data=True,
                                min_cluster_size=5,
                                )
    # labels = clusterer.fit(xmap_embedding.astype(np.float64)).labels_

    clusterer.fit(np.array(distances).astype(np.float64))

    return clusterer


def sample_and_measure(reference_sample_map,
                       moving_xmap,
                       parameters,
                       ):
    sample_map = sample(moving_xmap,
                  parameters,
                  )
    distance = np.mean(np.abs(reference_sample_map - sample_map))

    return distance


def align_map(reference_dataset, moving_dataset, reference_centre, moving_centre, delta=3):
    bounds = ((moving_centre[0] - delta, moving_centre[0] + delta),
              (moving_centre[1] - delta, moving_centre[1] + delta),
              (moving_centre[2] - delta, moving_centre[2] + delta),
              (0, 2 * np.pi),
              (0, 2 * np.pi),
              (0, 2 * np.pi),
              )

    reference_xmap = PanDDAXMap.from_dataset(reference_dataset)

    moving_xmap = PanDDAXMap.from_dataset(moving_dataset)

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

    print("\t\tResult parameters: {}".format(result.x))
    print("\t\tResult parameters: {}".format(result.fun))

    aligned_map = sample(moving_xmap,
                         result.x,
                         )

    return aligned_map, result.fun


def align_maps(reference_dataset, datasets, alignments):
    results = []
    for dtag, dataset in datasets.items():
        print("\t\tAligning map {}".format(dtag))
        result = align_map(reference_dataset,
                           dataset,
                           alignments[reference_dataset.id],
                           alignments[dtag],
                           )

        results.append(result)

        # map_dict(lambda x: align_map(reference_dataset, x),
        #                datasets,
        #                )

    xmaps = [result[0] for result in results]
    distances = [result[1] for result in results]

    return xmaps, distances


def get_reference_dataset(datasets):
    reference_dataset = min(list(datasets.values()),
                            key=lambda x: x.get_resolution_high(),
                            )
    return reference_dataset


def more_that_one_cluster(clusters):
    clusters_non_negative = clusters[clusters >= 0]
    if len(np.unique(clusters)) == 0:
        return False
    if len(np.unique(clusters)) == 1:
        return False
    if len(np.unique(clusters)) > 1:
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

    reference_cluster = cluster_distances[arg]

    unclustered_datasets = {dtag: truncated_datasets[dtag]
                            for i, dtag
                            in enumerate(truncated_datasets)
                            if cluster_distances[i] == reference_cluster
                            }

    return unclustered_datasets


def cluster_datasets(truncated_datasets,
                     residues,
                     ):

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

    cluster_distances = cluster(distances)
    print("\tDiscovered {} unique clusters".format(np.unique(cluster_distances.labels_)))

    unclustered_datasets = get_unclustered_datasets(reference_dataset,
                                                    truncated_datasets,
                                                    cluster_distances,
                                                    )
    print("\tGot {} unclustered datasets".format(len(unclustered_datasets)))

    if more_that_one_cluster(cluster_distances):
        return [aligned_maps] + [x for x in cluster_datasets(unclustered_datasets, residues)]
    else:
        return [aligned_maps]


def get_comparable_residues(datasets,
                            residue_id,
                            ):
    residue_map = {}

    for dtag, dataset in datasets.items():
        try:
            residue = dataset.structure.structure[residue_id[0]][residue_id[1]][residue_id[2]]
        except Exception as e:
            print(e)
            residue = None

        residue_map[dtag] = residue

    return residue_map


def visualise_clusters(clusters,
                       path,
                       ):
    fig, axs = plt.subplots(nrows=len(clusters),
                            ncols=max([len(cluster) for cluster in clusters]),
                            )

    for i, cluster in enumerate(clusters):
        for j, xmap in enumerate(cluster):
            image = np.mean(xmap, axis=0)
            axs[i, j].imshow(image)

    fig.savefig(str(path))

def try_load(path):
    try:
        dataset = PanDDADataset.from_dir(path)
        return dataset
    except Exception as e:
        print(e)
        return None


def main():
    args = Args()

    fs = ClusterFSModel(args.data_dirs,
                        args.out_dir,
                        )

    datasets = map_dict(try_load,
                        fs.initial_model_dirs,
                        )
    datasets = {dtag: dataset for dtag, dataset in datasets.items() if dataset is not None}
    datasets = {item[0]: item[1] for i, item in enumerate(datasets.items()) if i < 60}
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

    for model in reference_dataset.structure.structure:
        for chain in model:
            for residue in chain:

                residue_id = (model.name, chain.name, residue.seqid.num)
                print("\tWorking on dataset: {}".format(residue_id))

                residues = get_comparable_residues(datasets,
                                                   residue_id,
                                                   )
                print("\tGot {} residues".format(len(residues)))
                residues = {dtag: residue for dtag, residue in residues.items() if residue is not None}
                print("\tGot {} residues".format(len(residues)))

                clusters = cluster_datasets(truncated_datasets,
                                            residues,
                                            )

                visualise_clusters(clusters,
                                   fs.output_dir / "{}.png".format(residue),
                                   )


if __name__ == "__main__":
    main()
