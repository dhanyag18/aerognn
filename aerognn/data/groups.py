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
    **{i: f"optimized_4{i}" for i in range(484, 499)}
    }
    
    return BATCH_GROUPS