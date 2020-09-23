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
    events: Events = Events.from_fs(fs,
                                    config.pandda_version,
                                    )
    print("\tGot models of {} events".format(len(events)))

    # Autobuild the events
    print("Autobuilding...")
    event_autobuilding_results: Dict[EventID, EventBuildResults] = {}
    for event_id in events:
        print(f"Processing event: {event_id}")
        event: Event = events[event_id]
        out_dir: EventDir = EventDir.from_event(event)
        initial_mtz_file: MtzFile = MtzFile.from_event(event)
        event_map_file: Ccp4File = Ccp4File.from_event(event)
        smiles_file: SmilesFile = SmilesFile.from_event(event)
        pdb_file: PdbFile = PdbFile.from_event(event)

        # Get ligand cif
        ligand_cif_file: CifFile = CifFile.from_smiles_file(smiles_file,
                                                            out_dir,
                                                            )

        # Trim pdb
        stripped_structure_file: PdbFile = PdbFile(
            event.event_dir / AUTOBUILD_STRIPPED_PDB.format(dtag=event.event_id.dtag.dtag,
                                                            event_idx=event.event_id.event_idx.event_idx,
                                                            )
        )

        structure: Structure = Structure.from_pdb_file(pdb_file)
        trimmed_structure: Structure = structure.strip(event, radius=5.0)
        stripped_structure_file.save(trimmed_structure)

        # Get masked event map
        event_map: Xmap = Xmap.from_file(event_map_file)
        masked_event_map: Xmap = event_map.mask_event(event, radius=7.0)

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
                                                      stripped_structure_file,
                                                      event,
                                                      )

        # Autobuilding results
        event_autobuilding_results[event_id] = EventBuildResults.from_rhofit_dir(rhofit_dir,
                                                                                 event,
                                                                                 )

    # Organise the results
    autobuilding_results: AutobuildingResults = AutobuildingResults.from_event_build_results(event_autobuilding_results)

    # Get the best autobuild for each event
    best_event_builds: AutobuildingResults = autobuilding_results.get_best_event_builds()

    # TO Json
    best_event_builds.to_json(fs.autobuilding_results_table)


if __name__ == "__main__":
    main()
