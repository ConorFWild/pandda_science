class PanDDAFilesystemModel:
    def __init__(self, pandda_root_dir):
        self.pandda_root_dir = pandda_root_dir

        self.pandda_analyse_dir = pandda_root_dir / "analyses"
        self.pandda_inspect_events_path = self.pandda_analyse_dir / "pandda_inspect_events.csv"
        self.autobuilding_results_table = self.pandda_analyse_dir / "autobuilding_results.csv"

        self.pandda_processed_datasets_dir = pandda_root_dir / "processed_datasets"
        self.pandda_processed_datasets_dirs = list(self.pandda_analyse_dir.glob("*"))
