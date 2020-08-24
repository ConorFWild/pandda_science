from __future__ import annotations

import dataclasses

from pathlib import Path

import argparse

import gemmi


@dataclasses.dataclass()
class Config:
    xmap_in: Path
    mtz_out: Path

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

        args = parser.parse_args()

        return Config(xmap_in=Path(args.xmap_in),
                      mtz_out=Path(args.mtz_out),
                      )


def main():
    config = Config.from_config()

    m = gemmi.read_ccp4_map(str(config.xmap_in))
    print(m.grid.spacegroup)
    # m.setup()
    sf = gemmi.transform_map_to_f_phi(m.grid, half_l=True)
    print(sf.spacegroup)
    # data = sf
    print(dir(sf))
    data = sf.prepare_asu_data()

    mtz = gemmi.Mtz(with_base=True, dmin=2.0)
    # mtz = gemmi.Mtz()
    print(dir(mtz))
    mtz.spacegroup = sf.spacegroup
    # mtz.set_cell_for_all(sf.unit_cell)
    mtz.cell = sf.unit_cell
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data)
    mtz.write_to_file(str(config.mtz_out))

if __name__ == "__main__":
    main()