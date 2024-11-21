RS_TASKS_OBJ = {
    "MoveTo": ["cube"],
    "Stack": ["cubeA", "cubeB"],
    "Lift": ["cube"],
}

MS_TASKS_OBJ = {
    "MoveTo": ["cube"],
    "Stack": ["cubeA", "cubeB"],
    "Lift": ["cube"],
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
    "bin-picking": [["objA"]]
}

DMC_TASKS_OBJ = {
    "walker_run": ["leg"],
    "reacher_hard": ["target"],   
    "reacher_easy": ["target"],   
    "manipulator_bring_ball": ["target_ball"], 
    "manipulator_bring_peg": [["target_blade", "target_guard", "target_pommel"]] 
}

DMC_TASKS_PROMPT = {
    "walker_run": ["text", "leg"],
    "reacher_hard": ["text", "small orange ball"],
    "reacher_easy": ["text", "small orange ball"],
    "manipulator_bring_ball": ["text", "small orange ball"],
    "manipulator_bring_peg": ["text", "orange sword"] 
}

RS_TASKS_PROMPT = {
    "MoveTo": ["text", "a red cube"],
    "Stack": ["text", "a red cube and a green cube"],
    "Lift": ["text", "a red cube"],
}

MS_TASKS_PROMPT = {
    "MoveTo": ["text", "a red cube"],
    "Stack": ["text", "a red cube and a green cube"],
    "CustomStack": ["text", "a red cube and a green cube"],
    "Lift": ["text", "a red cube"],
    "TurnFaucet": ["text", "a faucet"],
    "PickSingleYCB": ["text", "a banana"],
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
    "bin-picking": ["text", "a green cube in a red bin"]
}

PRIMAL_TASKS = {
    "rs": "Stack",
    "ms": "Stack",
    "mw": "pick-place",
    "dmc": "walker_run"
}
