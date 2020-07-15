from data import GetMoleculesData


def main():
    search = GetMoleculesData()
    print(search)

    id_list = search.get_target_ids(target='NUDT5A')
    print(id_list)

    url = search.set_molecule_url(target_id=id_list[0])
    print(url)


    # results = get_molecules_json(url=url)
    # print(results)


    results_table = search.get_all_mol_responses()
    print(results_table)

if __name__ == "__main__":
    
