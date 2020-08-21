from autobuilding_paper.lib import ReferenceStructures, ResidueEventDistance, Ligands, PanDDAEvents


class PanDDAEventDistances:
    def __init__(self, distances):
        self.distances = distances

    @staticmethod
    def from_dir(pandda_dir):
        reference_structures = ReferenceStructures.from_dir(pandda_dir)
        print("\tGot {} reference structures".format(len(reference_structures)))

        events = PanDDAEvents.from_dir(pandda_dir)

        distances = {}
        for reference_dtag, reference_structure in reference_structures.to_dict().items():
            print("\t\tLooking for matches to {}".format(reference_dtag))

            event_distances = []
            for event_dtag, event in events.to_dict().items():
                # print("\t\t\tLooking for match in {}".format(event_dtag))
                if event_dtag.dtag != reference_dtag.dtag:
                    continue

                print(reference_structure)
                ligs = Ligands.from_structure(reference_structure)

                ligand_distances = []
                for lig in ligs.ligands:

                    ligand_distance = (ResidueEventDistance.from_residue_event(lig,
                                                                        event,
                                                                        )
                                ).to_float()

                    ligand_distances.append(ligand_distance)

                ligand_distance = min(ligand_distances)

                event_distances.append(ligand_distance)

            if len(event_distances) == 0:
                distances[reference_dtag] = 0
            else:
                distances[reference_dtag] = min(event_distances)

        return PanDDAEventDistances(distances)
