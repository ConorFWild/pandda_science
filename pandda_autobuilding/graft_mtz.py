import numpy as np

import gemmi

def array_to_index(event_mtz):
    event_data = np.array(event_mtz, copy=False)
    array_to_index_map = {}
    for i in range(event_data.shape[0]):
        h = int(event_data[i, 0])
        k = int(event_data[i, 1])
        l = int(event_data[i, 2])
        array_to_index_map[i] = (h, k, l)

    return array_to_index_map


def index_to_array(intial_mtz):
    event_data = np.array(intial_mtz, copy=False)
    index_to_array_map = {}
    for i in range(event_data.shape[0]):
        h = int(event_data[i, 0])
        k = int(event_data[i, 1])
        l = int(event_data[i, 2])
        index_to_array_map[(h, k, l)] = i

    return index_to_array_map


def phase_graft(initial_mtz_path,
                event_mtz_path,
                out_path,
                ):
    intial_mtz = gemmi.read_mtz_file(str(initial_mtz_path))
    event_mtz = gemmi.read_mtz_file(str(event_mtz_path))

    array_to_index_map = array_to_index(intial_mtz)
    index_to_array_map = index_to_array(event_mtz)

    initial_mtz_data = np.array(intial_mtz, copy=False)
    event_mtz_data = np.array(event_mtz, copy=False)
    # print(initial_mtz_data.shape)
    # print(event_mtz_data.shape)

    # FWT
    initial_mtz_fwt = intial_mtz.column_with_label('FWT')
    # initial_mtz_fwt_index = initial_mtz_fwt.dataset_id
    initial_mtz_fwt_index = intial_mtz.column_labels().index("FWT")

    event_mtz_fwt = event_mtz.column_with_label('FWT')
    event_mtz_fwt_index = event_mtz.column_labels().index("FWT")

    # print("\t{}, {}".format(initial_mtz_data.shape, event_mtz_data.shape))
    # print(list(array_to_index_map.keys())[:10])
    # print(list(index_to_array_map.keys())[:10])

    skipped = 0
    for intial_array in range(initial_mtz_data.shape[0]):
        try:
            index = array_to_index_map[intial_array]
            event_array = index_to_array_map[index]
            initial_mtz_data[intial_array, initial_mtz_fwt_index] = event_mtz_data[event_array, event_mtz_fwt_index]
        except Exception as e:
            skipped = skipped + 1
            initial_mtz_data[intial_array, initial_mtz_fwt_index] = 0
    intial_mtz.set_data(initial_mtz_data)
    print("\tSkipped {} reflections".format(skipped))

    # PHWT
    initial_mtz_phwt = intial_mtz.column_with_label('PHWT')
    # initial_mtz_phwt_index = initial_mtz_phwt.dataset_id
    initial_mtz_phwt_index = intial_mtz.column_labels().index("PHWT")


    event_mtz_phwt = event_mtz.column_with_label('PHWT')
    # event_mtz_phwt_index = event_mtz_phwt.dataset_id
    event_mtz_phwt_index = event_mtz.column_labels().index("PHWT")


    skipped = 0
    for intial_array in range(initial_mtz_data.shape[0]):
        try:
            index = array_to_index_map[intial_array]
            event_array = index_to_array_map[index]
            initial_mtz_data[intial_array, initial_mtz_phwt_index] = event_mtz_data[event_array, event_mtz_phwt_index]
        except Exception as e:
            skipped = skipped + 1
            initial_mtz_data[intial_array, initial_mtz_phwt_index] = 0
    intial_mtz.set_data(initial_mtz_data)
    print("\tCopied FWT from {} to {}".format(event_mtz_fwt_index, initial_mtz_fwt_index))
    print("\tCopied PHWT from {} to {}".format(event_mtz_phwt_index, initial_mtz_phwt_index))
    print("\tSkipper {} reflections".format(skipped))

    intial_mtz.write_to_file(str(out_path))
