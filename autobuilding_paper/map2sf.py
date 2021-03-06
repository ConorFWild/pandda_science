from __future__ import annotations

import dataclasses

from pathlib import Path

import argparse

import gemmi


@dataclasses.dataclass()
class Config:
    xmap_in: Path
    mtz_out: Path
    resolution: float

    @staticmethod
    def from_config():
        parser = argparse.ArgumentParser()
        # IO
        parser.add_argument("-i", "--xmap_in",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )
        parser.add_argument("-o", "--mtz_out",
                            type=str,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )
        parser.add_argument("-r", "--resolution",
                            type=float,
                            help="The directory OF THE ROOT OF THE XCHEM DATABASE",
                            required=True
                            )

        args = parser.parse_args()

        return Config(xmap_in=Path(args.xmap_in),
                      mtz_out=Path(args.mtz_out),
                      resolution=float(args.resolution)
                      )


def main():
    config = Config.from_config()

    m = gemmi.read_ccp4_map(str(config.xmap_in))
    print(dir(m))
    # m.spacegroup = gemmi.find_spacegroup_by_name('P1')
    print(m.grid.spacegroup)
    m.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    print(m.grid.spacegroup)


    m.setup()
    print(m.grid.spacegroup)

    m.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
    print(m.grid.spacegroup)
    sf = gemmi.transform_map_to_f_phi(m.grid, half_l=True)
    print(sf.spacegroup)
    # data = sf
    print(dir(sf))
    data = sf.prepare_asu_data(dmin=config.resolution, with_000=True)

    mtz = gemmi.Mtz(with_base=True)
    # mtz = gemmi.Mtz()
    print(dir(mtz))
    mtz.spacegroup = sf.spacegroup
    # mtz.spacegroup = gemmi.find_spacegroup_by_name('P1')
    # mtz.set_cell_for_all(sf.unit_cell)
    mtz.cell = sf.unit_cell
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data)
    mtz.write_to_file(str(config.mtz_out))

if __name__ == "__main__":
    main()