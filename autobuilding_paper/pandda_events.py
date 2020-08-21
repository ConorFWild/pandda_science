from autobuilding_paper.lib import ReferenceStructures, ResidueEventDistance, Ligands, PanDDAEvents


class PanDDAEventDistances:
    def __init__(self, distances):
        self.distances = distances

    @staticmethod
    def from_dir(pandda_dir):
        reference_structures = ReferenceStructures.from_dir(pandda_dir)
        print("Got {} reference structures".format(len(reference_structures)))

        events = PanDDAEvents.from_dir(pandda_dir)

        distances = {}
        for reference_dtag, reference_structure in reference_structures.to_dict().items():
            print("\tLooking for matches to {}".format(reference_dtag))

            event_distances = []
            for event_dtag, event in events.to_dict().items():
                if event_dtag != reference_dtag:
                    continue

                lig = Ligands.from_structure(reference_structure)

                distance = (ResidueEventDistance.from_residue_event(lig,
                                                                    event,
                                                                    )
                            ).to_float()

                event_distances.append(distance)

            if len(event_distances) == 0:
                distances[reference_dtag] = 0
            else:
                distances[reference_dtag] = min(event_distances)

        return PanDDAEventDistances(distances)
