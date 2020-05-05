import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
import bokeh.models as bmo
from bokeh.palettes import d3
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap

from mdc.dataset import PanDDADataset
from mdc.xmap import PanDDAXMap




class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-r", "--root_path",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        parser.add_argument("-o", "--out_dir",
                            type=str,
                            help="The directory for output and intermediate files to be saved to",
                            required=True
                            )

        parser.add_argument("-n", "--n_procs",
                            type=str,
                            help="Number of processes to start",
                            required=True
                            )

        parser.add_argument("-c", "--clean_run",
                            type=bool,
                            help="Number of processes to start",
                            default=True,
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

        self.root_path = Path(args.root_dir)
        self.out_dir = Path(args.out_dir)
        self.n_procs = int(args.n_procs)
        self.clean_run = bool(args.clean_run)
        self.pdb_regex = str(args.pdb_regex)
        self.mtz_regex = str(args.mtz_regex)
        self.structure_factors = str(args.structure_factors)
        self.align = bool(args.align)
        self.data_dirs = Path(args.data_dirs)
        self.out_dir = Path(args.out_dir)
        self.processor = str(args.processor)
        self.n_procs = int(args.n_procs)


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


import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def embed(numpy_maps):
    # Convert to sample by feature
    sample_by_feature = np.vstack(numpy_maps)

    # Reduce dimension by PCA
    pca = PCA(n_components=min(50, len(numpy_maps)))
    sample_by_feature_pca = pca.fit_transform(sample_by_feature)

    # Reduce dimension by TSNE
    tsne = TSNE(n_components=2,
                method="exact")
    sample_by_feature_tsne = tsne.fit_transform(sample_by_feature_pca)

    return sample_by_feature_tsne


def cluster(xmap_embedding):
    # Perform initial clustering
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True,
                                prediction_data=True,
                                min_cluster_size=5,
                                )
    # labels = clusterer.fit(xmap_embedding.astype(np.float64)).labels_

    clusterer.fit(xmap_embedding.astype(np.float64))

    return clusterer


def make_table(embedding,
               clustering,
               ):
    pass


def make_clustering_html(embedding,
                         clustering,
                         cluster_html_path,
                         ):
    pass


def main():
    args = Args()

    fs = ClusterFSModel(args.data_dirs,
                        args.out_dir,
                        )

    datasets = map_dict(PanDDADataset.from_dir,
                        fs.initial_model_dirs,
                        )

    datasets_res_high = min([x.get_resolution() for x in datasets.values()])

    truncated_datasets = map_dict(lambda x: x.mtz.truncate_reflections(datasets_res_high),
                                  datasets,
                                  )

    xmaps = map_dict(PanDDAXMap.from_dataset,
                     truncated_datasets,
                     )

    arrays = map_dict(np.array,
                      xmaps,
                      )

    flat_arrays = map_dict(lambda x: x.flatten(),
                           arrays,
                           )

    embedding = embed(flat_arrays)

    clustering = cluster(embedding)

    table = make_table(embedding,
                       clustering,
                       )

    make_clustering_html(embedding,
                         clustering,
                         fs.cluster_html_path,
                         )

    table.to_csv(str(fs.cluster_table_path))


if __name__ == "__main__":
    main()
