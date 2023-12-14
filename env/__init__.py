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
    "robobin": [["cubeA", "cubeB"]],
}

DMC_TASKS_OBJ = {
    "reacher_hard": ["target"],   
    "reacher_easy": ["target"],   
    "manipulator_bring_ball": ["target_ball"], 
    "manipulator_bring_peg": [["target_blade", "target_guard", "target_pommel"]] 
}

DMC_TASKS_PROMPT = {
    "walker_run": ["text", "leg"],
    "reacher_hard": ["text", "small orange ball"],
    "reacher_easy": ["text", "small orange ball"],
    "manipulator_bring_ball": ["text", "small green ball", "orange robot"],
    "manipulator_bring_peg": ["text", "orange sword"] 
}

RS_TASKS_PROMPT = {
    "MoveTo": ["text", "a red cube"],
    "Stack": ["text", "a red cube and a green cube"],
    "CustomStack": ["text", "a red cube and a green cube"],
    "Lift": ["text", "a red cube"],
    "CustomLift": ["text", "a red cube"],
}

MS_TASKS_PROMPT = {
    "MoveTo": ["text", "a red cube"],
    "Stack": ["text", "a red cube and a green cube"],
    "CustomStack": ["text", "a red cube and a green cube"],
    "Lift": ["text", "a red cube"],
    "CustomLift": ["text", "a red cube"],
    "TurnFaucet": ["text", "a faucet"],
    "PickSingleYCB": ["text", "a banana"],
    "CustomLiftYCB": ["text", "a banana"],
}

MW_TASKS_PROMPT = {
    "drawer-close": ["text", "a green drawer"],
    "drawer-open": ["text", "a green drawer"],
    "disassemble": ["text", "a green nut"],
    "shelf-place": ["text", "a blue cube"],
    "handle-pull": ["text", "a red handle"],
    "door-open": ["text", "a grey door"],
    "door-close": ["text", "a grey door"],
    "peg-insert-side": ["text", "a green peg"],
    "hammer": ["text", "hammer"],
    "robobin": ["text", "green cube"],
}

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS

PRIMAL_TASKS = {
    "walker": "walker_stand",
    "jaco": "jaco_reach_top_left",
    "quadruped": "quadruped_walk",
    "rs": "Stack",
    "ms": "Stack",
    "mw": "pick-place",
    "dmc": "walker_run"
}
