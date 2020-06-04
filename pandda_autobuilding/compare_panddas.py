from pandda_autobuilding.pandda_file_system_model import PanDDAFilesystemModel
from pandda_autobuilding.get_events import get_events
from pandda_autobuilding.dist import map_parallel
from pandda_autobuilding.autobuild_event import autobuild_event

RHOFIT_CONVENTIONAL_DIR = "rhofit_conventional"

def make_rscc_table(results_1, results_2,):
    records = []
    for event_id in results_1:
        record = {}
        record["conventional_rscc"] = results_1[event_id].rscc
        record["event_rscc"] = results_2[event_id].rscc
        records.append(record)

    return pd.DataFrame(records)

def main():
    # # Get config
    config = Config()

    # # Get file system model: Dir<pandda> -> FSModel
    print("Building I/O model...")
    fs_in = PanDDAFilesystemModel(config.input_pandda_dir)
    print("\tFound {} dataset dirs".format(len(fs.pandda_processed_datasets_dirs)))

    # # Get output Model
    fs_out = ComparisonFileSystemModel(config.out_dir)

    # # Get events: Path<inpsect table> -> Dict<Tuple<Dtag, Idx>, Event>
    print("Getting event models...")
    events = get_events(fs_in.pandda_inspect_events_path)
    print("\tGot models of {} events".format(len(events)))

    # # Autobuild 2fofc:
    print("Autobuilding...")
    conventional_results = map_parallel(autobuild_event,
                                        events,
                                        )

    # # Autobuild PanDDA event
    event_results = map_parallel(delayed(autobuild_event_rhofit)(event.mtz_path,
                                                                 event.ligand_path,
                                                                 event.pdb_path,
                                                                 event.coords,
                                                                 fs_out[(event.dtag,
                                                                         event.event_idx)] / RHOFIT_CONVENTIONAL_DIR,
                                                                 )
                                 for event
                                 in events
                                 )

    # # Make table
    rscc_table = make_rscc_table(conventional_results,
                                 event_results,
                                 )
    rscc_table.to_csv(fs_out.rscc_table_path)

    # # Make scatter
    save_scatter(rscc_table,
                 "2fofc",
                 "event",
                 fs_out.rscc_scatter_path,
                 )



if __name__ == "__main__":
    main()