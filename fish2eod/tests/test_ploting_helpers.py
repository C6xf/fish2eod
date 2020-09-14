from itertools import combinations
from tempfile import TemporaryDirectory

import numpy as np

from fish2eod.analysis.plotting import generate_mask
from fish2eod.models import BaseFishModel
from fish2eod.xdmf.load import load_from_file
from fish2eod.xdmf.save import Saver


def test_mask_single():
    m = BaseFishModel()
    m.compile(fish_x=[0, 20], fish_y=[0, 0])
    m.solve(fish_x=[0, 20], fish_y=[0, 0])

    d = TemporaryDirectory()
    Saver("m", d.name).save_model(m)

    load_handle = load_from_file(f"{d.name}/m")

    t, g, v = load_handle["domain"].load_data()
    for domain_name, domain_id in load_handle.domain_map.items():
        mask = generate_mask(solution=load_handle, include_domains=(domain_name,))
        included_value = np.unique(v[~np.array(mask)])
        assert len(included_value) == 1
        assert included_value[0] == domain_id


def test_mask_2_comb():  # todo dont use fish domains
    m = BaseFishModel()
    m.compile(**{"fish_x": [0, 20], "fish_y": [0, 0]})
    m.solve(**{"fish_x": [0, 20], "fish_y": [0, 0]})

    d = TemporaryDirectory()
    Saver("m", d.name).save_model(m)

    load_handle = load_from_file(f"{d.name}/m")

    t, g, v = load_handle["domain"].load_data()

    names = load_handle.domain_map.keys()
    for d1, d2 in combinations(names, 2):
        mask = generate_mask(solution=load_handle, include_domains=(d1, d2))
        included_value = np.unique(v[~np.array(mask)])
        assert len(included_value) == 2
        assert set(included_value) == {
            load_handle.domain_map[d1],
            load_handle.domain_map[d2],
        }
