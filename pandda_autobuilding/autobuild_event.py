import os

from pandda_autobuilding.event_map_to_mtz import event_map_to_mtz
from pandda_autobuilding.elbow import elbow
from pandda_autobuilding.strip_ligands import strip_protein
from pandda_autobuilding.graft_mtz import phase_graft



def autobuild_event(event):
    # Event map mtz
    print("\tMaking event map mtz...")
    initial_event_mtz_path = event.pandda_event_dir / "{}_{}.mtz".format(event.dtag, event.event_idx)
    try:
        os.remove(str(initial_event_mtz_path))
    except:
        pass

    formatted_command, stdout, stderr = event_map_to_mtz(event.event_map_path,
                                                         initial_event_mtz_path,
                                                         event.analysed_resolution,
                                                         )

    # Ligand cif
    print("\tMaking ligand cif...")
    ligand_path = event.pandda_event_dir / "ligand.cif"
    ligand_smiles_path = get_ligand_smiles(event.pandda_event_dir)
    if not ligand_path.exists():
        elbow(event.pandda_event_dir,
              ligand_smiles_path,
              )

    # Stripped protein
    print("\tStripping ligands near event...")
    intial_receptor_path = event.pandda_event_dir / "receptor_{}.pdb".format(event.event_idx)
    if not intial_receptor_path.exists():
        strip_protein(event.receptor_path,
                      event.coords,
                      intial_receptor_path,
                      )

    # Quick refine
    event_mtz_path = event.pandda_event_dir / "grafted_{}.mtz".format(event.event_idx)
    try:
        os.remove(str(event_mtz_path))
    except:
        pass
    if not event_mtz_path.exists():
        phase_graft(event.initial_mtz_path,
                    initial_event_mtz_path,
                    event_mtz_path,
                    )

    if not event_mtz_path.exists():
        raise Exception("Could not find event mtz after attempting generation: {}".format(event_mtz_path))

    if not ligand_path.exists():
        raise Exception("Could not find ligand cif path after attempting generation: {}".format(event_mtz_path))

    # if not receptor_path.exists():
    #     raise Exception("Could not find event receptor path after attempting generation: {}".format(event_mtz_path))

    out_dir_path = event.pandda_event_dir / "rhofit_{}".format(event.event_idx)

    try:
        shutil.rmtree(str(out_dir_path))
    except:
        pass

    # os.mkdir(str(out_dir_path))

    # autobuilding_command = AutobuildingCommand(out_dir_path=out_dir_path,
    #                                            mtz_path=event_mtz_path,
    #                                            ligand_path=event.ligand_path,
    #                                            receptor_path=receptor_path,
    #                                            coord=event.coords,
    #                                            )
    #
    # formatted_command, stdout, stderr = execute(autobuilding_command)
    #
    # autobuilding_log_path = out_dir_path / "pandda_autobuild_log.txt"
    # write_autobuild_log(formatted_command, stdout, stderr, autobuilding_log_path)

    print("\tAutobuilding...")
    # if (out_dir_path / "results.txt").exists():
    autobuilding_command = AutobuildingCommandRhofit(out_dir_path=out_dir_path,
                                                     mtz_path=event_mtz_path,
                                                     ligand_path=ligand_path,
                                                     receptor_path=intial_receptor_path,
                                                     )
    print("\t\tCommand: {}".format(str(autobuilding_command)))
    formatted_command, stdout, stderr = execute(autobuilding_command)

