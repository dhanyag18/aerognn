def get_groups():
    BATCH_GROUPS = {
        **{i: f"rand_{i}" for i in range(1, 72)},
        **{i: f"rand_{i}" for i in range(105, 126)},
        **{i: "explore_setback" for i in range(77, 82)},
        **{i: "explore_m" for i in range(82, 87)},
        **{i: "explore_bulge" for i in range(87, 91)},
        **{i: "explore_chamfer" for i in range(91, 95)},
        **{i: f"explore_{i}" for i in [72, 73, 74, 75, 76] + list(range(95, 105))},
        **{i: "old_gp_grid" for i in range(126, 135)},
        **{i: "old_xgb_grid" for i in range(135, 145)},
        **{i: "grid_batch_1" for i in range(145, 155)},
        **{i: "grid_batch_2" for i in range(155, 165)},
        **{i: "grid_batch_3" for i in range(165, 175)},
        **{i: "grid_batch_4" for i in range(175, 190)},
        **{i: "grid_batch_5" for i in range(190, 200)},
        **{i: "grid_batch_6" for i in range(200, 210)},
        **{i: f"de_gp_{i}" for i in range(210, 215)},
        **{i: f"de_xgb_{i}" for i in range(215, 220)},
        **{i: f"val_{i}" for i in range(220, 230)},
        **{i: f"xgb_opt_{i}" for i in range(230, 235)},
        **{i: f"gp_opt_{i}" for i in range(235, 240)},
        **{i: f"batch7_{i}" for i in range(240, 260)},
        **{i: f"batch8_{i}" for i in range(260, 295)},
        **{i: f"diverse_exploration_{i}" for i in range(295, 306)},
        **{i: f"diverse_exploration_2{i}" for i in range(306, 323)},
        **{i: f"diverse_exploration_3{i}" for i in range(323, 439)},
        **{i: f"optimized{i}" for i in range(439, 454)},
        **{i: f"optimized_2{i}" for i in range(454, 469)},
        **{i: f"optimized_3{i}" for i in range(469, 484)},
        **{i: f"optimized_4{i}" for i in range(484, 499)},
        **{i: f"active_learning_{i}" for i in range(499, 519)},
        **{i: f"active_learning_2_{i}" for i in range(519, 549)},
    }
    return BATCH_GROUPS


def get_coarse_groups():
    ALL_GROUPS = {
        **{i: f"rand_{i}" for i in range(1, 72)},
        **{i: f"rand_{i}" for i in range(105, 126)},
        **{i: "explore_setback" for i in range(77, 82)},
        **{i: "explore_m" for i in range(82, 87)},
        **{i: "explore_bulge" for i in range(87, 91)},
        **{i: "explore_chamfer" for i in range(91, 95)},
        **{i: f"explore_{i}" for i in [72, 73, 74, 75, 76] + list(range(95, 105))},
        **{i: "old_gp_grid" for i in range(126, 135)},
        **{i: "old_xgb_grid" for i in range(135, 145)},
        **{i: "grid_batch_1" for i in range(145, 155)},
        **{i: "grid_batch_2" for i in range(155, 165)},
        **{i: "grid_batch_3" for i in range(165, 175)},
        **{i: "grid_batch_4" for i in range(175, 190)},
        **{i: "grid_batch_5" for i in range(190, 200)},
        **{i: "grid_batch_6" for i in range(200, 210)},
        **{i: f"de_gp_{i}" for i in range(210, 215)},
        **{i: f"de_xgb_{i}" for i in range(215, 220)},
        **{i: f"val_{i}" for i in range(220, 230)},
        **{i: f"xgb_opt_{i}" for i in range(230, 235)},
        **{i: f"gp_opt_{i}" for i in range(235, 240)},
        **{i: f"batch7_{i}" for i in range(240, 260)},
        **{i: f"batch8_{i}" for i in range(260, 295)},
        **{i: f"diverse_exploration_{i}" for i in range(295, 306)},
        **{i: f"diverse_exploration_2{i}" for i in range(306, 323)},
        **{i: f"diverse_exploration_3{i}" for i in range(323, 439)},
        **{i: f"optimized{i}" for i in range(439, 454)},
        **{i: f"optimized_2{i}" for i in range(454, 469)},
        **{i: f"optimized_3{i}" for i in range(469, 484)},
        **{i: f"optimized_4{i}" for i in range(484, 499)},
        **{i: f"active_learning_{i}" for i in range(499, 519)},
        **{i: f"active_learning_2_{i}" for i in range(519, 549)},
    }

    fine_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        126, 277, 288, 135, 303, 304, 305, 306, 307, 308,
        309, 310, 311, 483, 484, 494, 495, 496, 410, 411,
        412, 413, 238, 239, 55, 56, 57, 58, 59, 60,
        61, 98, 266, 382, 383, 384, 385, 386, 158, 121,
    ]

    coarse_selected = {
        coarse_id: ALL_GROUPS[fine_id]
        for coarse_id, fine_id in enumerate(fine_ids, start=1)
    }

    coarse_random = {i: f"coarse_rand_{i}" for i in range(51, 201)}

    coarse_active = {
        **{i: "al1_iter1" for i in range(201, 206)},
        **{i: "al1_iter2" for i in range(206, 211)},
        **{i: "al1_iter3" for i in range(211, 216)},
        **{i: "al1_iter4" for i in range(216, 221)},
    }

    coarse_active_2 = {
        **{i: "al2_iter1" for i in range(221, 226)},
        **{i: "al2_iter2" for i in range(226, 231)},
        **{i: "al2_iter3" for i in range(231, 236)},
        **{i: "al2_iter4" for i in range(236, 241)},
        **{i: "al2_iter5" for i in range(241, 246)},
        **{i: "al2_iter6" for i in range(246, 251)},
    }
    coarse_active_3 = {
        **{i: "al3_iter1" for i in range(261, 266)}
    }
    coarse_active_4 = {
        **{i: "al4_iter1" for i in range(266, 271)}
    }

    coarse_typical = {
        251: ALL_GROUPS[72],
        252: ALL_GROUPS[73],
        253: ALL_GROUPS[77],
        254: ALL_GROUPS[104],
        255: ALL_GROUPS[105],
        256: ALL_GROUPS[107],
        257: ALL_GROUPS[67],
        258: ALL_GROUPS[149],
        259: ALL_GROUPS[167],
        260: ALL_GROUPS[233],
    }

    return {**coarse_selected, **coarse_random, **coarse_active, **coarse_active_2, **coarse_typical, **coarse_active_3, **coarse_active_4}


def get_coarse_to_fine():
    original = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
        11: 126, 12: 277, 13: 288, 14: 135, 15: 303, 16: 304, 17: 305, 18: 306, 19: 307, 20: 308,
        21: 309, 22: 310, 23: 311, 24: 483, 25: 484, 26: 494, 27: 495, 28: 496, 29: 410, 30: 411,
        31: 412, 32: 413, 33: 238, 34: 239, 35: 55, 36: 56, 37: 57, 38: 58, 39: 59, 40: 60,
        41: 61, 42: 98, 43: 266, 44: 382, 45: 383, 46: 384, 47: 385, 48: 386, 49: 158, 50: 121,
        251: 72, 252: 73, 253: 77, 254: 104, 255: 105,
        256: 107, 257: 67, 258: 149, 259: 167, 260: 233,
    }

    active_learning = {
        coarse_id: fine_id
        for coarse_id, fine_id in zip(range(201, 221), range(499, 519))
    }

    active_learning_2 = {
        coarse_id: fine_id
        for coarse_id, fine_id in zip(range(221, 251), range(519, 549))
    }
    
    active_learning_3 = {
        coarse_id: fine_id
        for coarse_id, fine_id in zip(range(261, 266), range(549, 554))
    }
    active_learning_4 = {
        coarse_id: fine_id
        for coarse_id, fine_id in zip(range(266, 271), range(554, 559))
    }

    return {**original, **active_learning, **active_learning_2, **active_learning_3, **active_learning_4}