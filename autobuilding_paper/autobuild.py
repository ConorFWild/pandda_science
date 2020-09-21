from __future__ import annotations

from typing import *

from autobuilding_paper.autobuild_types import *


def main():
    # Get config
    config = Config()

    # Model Pandda file system
    print("Building I/O model...")
    fs = PanDDAFilesystemModel(config.input_pandda_dir)   
    print("\tFound {} dataset dirs".format(len(fs.pandda_processed_datasets_dirs)))

    # Get events
    print("Getting event models...")
    events: Events = Events.from_fs(fs.pandda_analyse_events_path,
                                    config.pandda_version,
                                    )   
    print("\tGot models of {} events".format(len(events)))

    # Autobuild the events
    print("Autobuilding...")
    event_autobuilding_results: Dict[EventID, EventBuildResults] = {}
    for event_id in events:
        event: Event = events[event_id]
        out_dir: EventDir = EventDir.from_event(event)   
        initial_mtz_file: MtzFile = MtzFile.from_event(event)
        event_map_file: Ccp4File = Ccp4File.from_event(event)
        smiles_file: SmilesFile = SmilesFile.from_event(event)   

        # Get ligand cif
        ligand_cif_file: CifFile = CifFile.from_smiles_file(smiles_file,
                                                            out_dir,
                                                            )   

        # Get masked event map
        event_map: Xmap = Xmap.from_file(event_map_file)   
        masked_event_map: Xmap = event_map.mask_event(event)   

        # Get event mtz
        initial_mtz: Reflections = Reflections.from_file(initial_mtz_file)
        naive_event_mtz: Reflections = Reflections.from_xmap(masked_event_map,
                                                             initial_mtz,
                                                             )   
        event_mtz: Reflections = initial_mtz.merge_mtzs(naive_event_mtz)   
        event_mtz_file: MtzFile = MtzFile(event.event_dir / AUTOBUILD_EVENT_MTZ.format(dtag=event.event_id.dtag.dtag,
                                                                                       event_idx=event.event_id.event_idx.event_idx,
                                                                                       ))
        event_mtz_file.save(event_mtz)

        # Autobuild
        rhofit_dir: RhofitDir = RhofitDir.from_rhofit(event_mtz_file,
                                                      ligand_cif_file,
                                                      event,
                                                      )

        # Autobuilding json
        event_autobuilding_results[event_id] = EventBuildResults.from_rhofit_dir(rhofit_dir,
                                                                                 event,
                                                                                 )

    # Organise the results
    autobuilding_results: AutobuildingResults = AutobuildingResults.from_event_build_results(event_autobuilding_results)

    # Get the best autobuild for each event
    best_event_builds: AutobuildingResults = autobuilding_results.get_best_event_builds()
    best_event_builds_dict = best_event_builds.to_flat_dict()


if __name__ == "__main__":
    main()
