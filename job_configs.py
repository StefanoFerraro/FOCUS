import os
from collections import defaultdict

if os.environ["EAI_ACCOUNT_ID"] == "7c445bd5-8d23-48dd-963e-b0480d238c8a":
    user = "pau"
elif os.environ["EAI_ACCOUNT_ID"] == "68fdc833-1eaa-4b60-9b64-ac27814e4b61":
    user = "sai"
elif os.environ["EAI_ACCOUNT_ID"] == "bf73ae4a-a155-4c56-b304-d4ccd60b37f1":
    user = "mazpie"

MNT_PUBLIC = defaultdict(lambda:"snow.colab.public")

MNT_HOME = {
    "sai"    : "snow.sai.home",
    "mazpie" : "snow.colab.mazpie_home"
}
REG_IMAGE = {
    "mazpie" : "mila.mazpie/focus:latest",
    "sai"    : "snow.sai/vcwang"
}
ACCOUNT_ID = {
    "mazpie" : "ee5d2daa-0d0f-47d9-80c6-350abb498c77", # "bf73ae4a-a155-4c56-b304-d4ccd60b37f1",
    "sai"    : "68fdc833-1eaa-4b60-9b64-ac27814e4b61"
}

WANDB_API_KEY = {
    "mazpie" : "9821d18d216a3a2e66c87759ccd1fde2c3c8d0ff", # "993646428610a189281ef60a615af52e1c4bec24",
    "sai"    : "6d3ca87335490cecbaafd6cfe982c22f63e23b10"
}

PYTHON_BINARIES = {
    "mazpie" : "/opt/conda/bin/python",
    "sai"    : "/mnt/home/trainenv/bin/python"
}

JOB_CONFIG = { user : 
    {
        "account_id": ACCOUNT_ID[user],
        "image": f"registry.console.elementai.com/{REG_IMAGE[user]}",
        "data": [f"{MNT_PUBLIC[user]}:/mnt/public", f"{MNT_HOME[user]}:/mnt/home"],
        "restartable": True,
        "preemptable" : True,
        "resources": {
            "cpu": 4,
            "mem": 64,
            "gpu": 1,
            "gpu_model": "V100",
            "gpu_mem" : 32,
        },
        "interactive": False,
        "bid": 0,
    }
for user in MNT_HOME }