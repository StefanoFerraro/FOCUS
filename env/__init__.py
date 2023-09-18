DOMAINS = [
    "walker",
    "quadruped",
    "jaco",
]

WALKER_TASKS = [
    "walker_stand",
    "walker_walk",
    "walker_run",
    "walker_flip",
]

QUADRUPED_TASKS = [
    "quadruped_walk",
    "quadruped_run",
    "quadruped_stand",
    "quadruped_jump",
]

JACO_TASKS = [
    "jaco_reach_top_left",
    "jaco_reach_top_right",
    "jaco_reach_bottom_left",
    "jaco_reach_bottom_right",
]

RS_TASKS_OBJ = {
    "MoveTo": ["cube"],
    "Stack": ["cubeA", "cubeB"],
    "CustomStack": ["cubeA", "cubeB"],
    "Lift": ["cube"],
    "CustomLift": ["cube"],
    "NutAssembly": ["nut"],
}

MS_TASKS_OBJ = {
    "MoveTo": ["cube"],
    "Stack": ["cubeA", "cubeB"],
    "CustomStack": ["cubeA", "cubeB"],
    "Lift": ["cube"],
    "CustomLift": ["cube"],
    "TurnFaucet": ["faucet"],
    "PickSingleYCB": ["obj"],
    "CustomLiftYCB": ["obj"],
}

MW_TASKS_OBJ = {
    "drawer-close": [["drawer", "drawer_link", "drawercase_link"]],
    "drawer-open": [["drawer", "drawer_link", "drawercase_link"]],
    "disassemble": [["RoundNut"]],
    "shelf-place": [["obj"]],
    "handle-pull": [["handle_link", "hdlprs"]],
    "door-open": [["door", "door_link"]],
    "door-close": [["door", "door_link"]],
    "peg-insert-side": [["peg"]],
    "hammer": [["hammerbody"]],
}

MW_TASKS_PROMPT = {
    "drawer-close": ["box", [50, 175, 225, 300]],
    "drawer-open": ["box", [50, 175, 225, 300]],
    "disassemble": ["box", [175, 325, 250, 375]],
    "shelf-place": ["box", [210, 320, 250, 350]],
    "handle-pull": ["box", [50, 250, 225, 350]],
    "door-open": ["box", [0, 100, 150, 350]],
    "door-close": ["box", [200, 225, 300, 300]],
    "peg-insert-side": ["box", [130, 330, 175, 375]],
    "hammer": ["text", "hammer"],
}

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS

PRIMAL_TASKS = {
    "walker": "walker_stand",
    "jaco": "jaco_reach_top_left",
    "quadruped": "quadruped_walk",
    "rs": "Stack",
    "ms": "Stack",
    "mw": "pick-place",
}
