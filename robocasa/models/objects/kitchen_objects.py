import torch
from copy import deepcopy
import pathlib
import os
import math
import random
import xml.etree.ElementTree as ET
import json
from collections import defaultdict

import numpy as np
from robosuite.utils.mjcf_utils import find_elements, string_to_array

import robocasa

ALL_OBJ_INFOS = json.load(open('/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/all_infos.json', 'r'))
rank_info = torch.load('/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/rank_info.pt', map_location='cpu')
OBJ_NAME_LIST = rank_info['obj_name_list']
ORI_RANK = rank_info['ori_rank']
OBJ_SIM_MATRIX = rank_info['obj_sim_matrix']
ATTR2IDX = {
    'color': 0,
    'shape': 1,
    'material': 2,
    'class': 3
}
CLASSNAME2IDX = {OBJ_NAME_LIST[idx]: idx for idx in range(len(OBJ_NAME_LIST))}


BASE_ASSET_ZOO_PATH = os.path.join(robocasa.models.assets_root, "objects")

# Constant that contains information about each object category. These will be used to generate the ObjCat classes for each category
OBJ_CATEGORIES = dict(
    liquor=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            model_folders=["aigen_objs/alcohol"],
            scale=1.50,
        ),
        objaverse=dict(
            model_folders=["objaverse/alcohol"],
            scale=1.35,
            exclude=[
                "alcohol_5",
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/alcohol"],
            scale=1.0
        )
    ),
    apple=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.90,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/apple"],
            scale=0.4,
            exclude=[
                "apple_34",
                "apple_29",
                "apple_8",
                "apple_7"
            ]
        )
    ),
    avocado=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/avocado"],
            scale=1.0
        )
    ),
    bagel=dict(
        types=("bread_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.2,
        ),
        objaverse=dict(
            exclude=[
                "bagel_8",
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/bagel"],
            scale=1.0,
        )
    ),
    bagged_food=dict(
        types=("packaged_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(
            exclude=[
                "bagged_food_12",
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/bagged_food"],
            scale=2.0,
            exclude=[
                "bagged_food_2"
            ]
        )
    ),
    baguette=dict(
        types=("bread_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "baguette_3",  # small holes on ends
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/baguette"],
            scale=2.0
        ),
    ),
    banana=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.10,
        ),
        objaverse=dict(
            scale=0.95,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/banana"],
            scale=1.0
        )
    ),
    bar=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[1.25, 1.25, 1.75],
        ),
        objaverse=dict(
            scale=[0.75, 0.75, 1.2],
            exclude=[
                "bar_1",  # small holes scattered
            ],
        )
    ),
    bar_soap=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=[1.25, 1.25, 1.40],
        ),
        objaverse=dict(
            scale=[0.95, 0.95, 1.05],
            exclude=["bar_soap_2"],
        ),
    ),
    beer=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(scale=1.15),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/beer"],
            scale=1.0,
            exclude=["beer_1"]
        )
    ),
    bell_pepper=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.75,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/bell_pepper"],
            scale=0.75,
        )
    ),
    bottled_drink=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/bottled_drink"],
            scale=1.0,
            exclude=[
                "bottled_drink_18"
            ]
        )
    ),
    bottled_water=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(
            scale=1.10,
            exclude=[
                "bottled_water_0",  # minor hole at top
                "bottled_water_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
        ),
    ),
    bowl=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=2.0,
            exclude=[
                "bowl_21",  # can see through from bottom of bowl
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/bowl"],
            scale=1.3,
            exclude=["bowl_53", "bowl_24", "bowl_15"]
        )
    ),
    boxed_drink=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(
            scale=0.80,
            exclude=[
                "boxed_drink_9",  # hole on bottom
                "boxed_drink_6",  # hole on bottom
                "boxed_drink_8",  # hole on bottom
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/boxed_drink"],
            scale=1.0,
            exclude=["boxed_drink_8"]
        )
    ),
    boxed_food=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.1,
            exclude=[
                "boxed_food_5",  # causing error. eigenvalues of mesh inertia violate A + B >= C
            ],
            # exclude=[
            #     "boxed_food_5",
            #     "boxed_food_3", "boxed_food_1", "boxed_food_6", "boxed_food_11", "boxed_food_10", "boxed_food_8", "boxed_food_9", "boxed_food_7", "boxed_food_2", # self turning due to single collision geom
            # ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/boxed_food"],
            scale=1.2
        )
    ),
    bread=dict(
        types=("bread_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[0.80, 0.80, 1.0],
        ),
        objaverse=dict(scale=[0.70, 0.70, 1.0], exclude=["bread_22"]),  # hole on bottom
        objaverse_extra=dict(
            model_folders=["objaverse_extra/bread"],
            scale=0.8,
        )
    ),
    broccoli=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                "broccoli_2",  # holes on one part
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/broccoli"],
            scale=0.6,
        )
    ),
    cake=dict(
        types=("sweets"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.8,
        ),
        objaverse=dict(
            scale=0.8,
            exclude=[
                "cake_2"
            ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cake"],
            scale=1.2,
            exclude=["cake_20"]
        )
    ),
    can=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(),
        objaverse=dict(
            exclude=[
                "can_17",
                "can_10", # hole on bottom
                "can_5", # causing error: faces of mesh have inconsistent orientation.
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/can"],
            scale=0.9,
            exclude=["can_41"]
        )
    ),
    candle=dict(
        types=("decoration"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            exclude=[
                "candle_11",  # hole at bottom
                # "candle_2", # can't see from bottom view angle
                # "candle_15", # can't see from bottom view angle
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/candle"],
            scale=1.0,
        )
    ),
    canned_food=dict(
        types=("packaged_food"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            scale=0.90,
            exclude=[
                "canned_food_7",  # holes at top and bottom
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/canned_food"],
            scale=0.9,
        )
    ),
    carrot=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/carrot"],
            scale=1.0,
        )
    ),
    cereal=dict(
        types=("packaged_food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            # exclude=[
            #     "cereal_2", "cereal_5", "cereal_13", "cereal_3", "cereal_9", "cereal_0", "cereal_7", "cereal_4", "cereal_8", "cereal_12", "cereal_11", "cereal_1", "cereal_6", "cereal_10", # self turning due to single collision geom
            # ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cereal"],
            scale=1.2,
        )
    ),
    cheese=dict( 
        types=("dairy"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.85,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cheese"],
            scale=0.6,
        )
    ),
    chips=dict(
        types=("packaged_food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            exclude=[
                "chips_12",  # minor hole at bottom of bag
                # "chips_2", # a weird texture at top/bottom but keeping this
            ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/chips"],
            scale=1.5,
        )
    ),
    chocolate=dict(
        types=("sweets"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=[1.0, 1.0, 1.35],
        ),
        objaverse=dict(
            scale=[0.80, 0.80, 1.20],
            exclude=[
                # "chocolate_2", # self turning due to single collision geom
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/chocolate"],
            scale=0.7,
        )
    ),
    coffee_cup=dict(
        types=("drink"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "coffee_cup_18",  # can see thru top
                "coffee_cup_5",  # can see thru from bottom side
                "coffee_cup_19",  # can see thru from bottom side
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/coffee_cup"],
            scale=0.8,
            exclude=["coffee_cup_44"]
        )
    ),
    condiment_bottle=dict(
        types=("condiment"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
            model_folders=["aigen_objs/condiment"],
        ),
        objaverse=dict(
            scale=1.05,
            model_folders=["objaverse/condiment"],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/condiment"],
            scale=1.0,
        )
    ),
    corn=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(scale=1.05),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/corn"],
            scale=1.0,
        )
    ),
    croissant=dict(
        types=("pastry"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/croissant"],
            scale=1.0,
        )
    ),
    cucumber=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cucumber"],
            scale=1.0,
        )
    ),
    cup=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cup"],
            scale=1.0,
            exclude=["cup_35"]
        )
    ),
    cupcake=dict(
        types=("sweets"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            exclude=[
                "cupcake_0",  # can see thru bottom
                "cupcake_10",  # can see thru bottom,
                "cupcake_1",  # very small hole at bottom
            ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cupcake"],
            scale=0.8,
        )
    ),
    cutting_board=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.0,
        ),
        objaverse=dict(
            scale=1.35,
            exclude=[
                "cutting_board_14",
                "cutting_board_3",
                "cutting_board_10",
                "cutting_board_6",  # these models still modeled with meshes which should work most of the time, but excluding them for safety
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cutting_board"],
            scale=2.0,
        )
    ),
    donut=dict(
        types=("sweets", "pastry"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.15,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/donut"],
            scale=0.9,
            exclude=["donut_19", "donut_21", "donut_33", "donut_34", "donut_35", "donut_5"]
        )
    ),
    egg=dict(
        types=("dairy"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            scale=0.85,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/egg"],
            scale=0.7,
        )
    ),
    eggplant=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(scale=0.95),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/eggplant"],
            scale=1.0,
        )
    ),
    fish=dict(
        types=("meat"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=[1.35, 1.35, 2.0],
        ),
        objaverse=dict(
            scale=[1.0, 1.0, 1.5],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/fish"],
            scale=1.0,
        )
    ),
    fork=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/fork"],
            scale=1.0,
        )
    ),
    garlic=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.3,
        ),
        objaverse=dict(scale=1.10, exclude=["garlic_3"]),  # has hole on side
        objaverse_extra=dict(
            model_folders=["objaverse_extra/garlic"],
            scale=1.0,
            exclude=["garlic_1"]
        )
    ),
    hot_dog=dict(
        types=("cooked_food"),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.4,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/hot_dog"],
            scale=1.0,
        )
    ),
    jam=dict(
        types=("packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(
            scale=0.90,
        )
    ),
    jug=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.5,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/jug"],
            scale=1.0,
        )
    ),
    ketchup=dict(
        types=("condiment"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "ketchup_5"  # causing error: faces of mesh have inconsistent orientation.
            ]
        ),
    ),
    kettle_electric=dict(
        types=("receptacle"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse=dict(
            scale=1.35,
            model_folders=["objaverse/kettle"],
            exclude=[
                f"kettle_{i}"
                for i in range(29)
                if i not in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        ),
        aigen=dict(
            scale=1.5,
            model_folders=["aigen_objs/kettle"],
            exclude=[f"kettle_{i}" for i in range(11) if i not in [0, 2, 6, 9, 10, 11]],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/kettle"],
            scale=1.3,
        )
    ),
    kettle_non_electric=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse=dict(
            scale=1.35,
            model_folders=["objaverse/kettle"],
            exclude=[
                f"kettle_{i}"
                for i in range(29)
                if i in [0, 7, 9, 12, 13, 17, 24, 25, 26, 27]
            ],
        ),
        aigen=dict(
            scale=1.5,
            model_folders=["aigen_objs/kettle"],
            exclude=[f"kettle_{i}" for i in range(11) if i in [0, 2, 6, 9, 10, 11]],
        ),
    ),
    kiwi=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=0.90,
        ),
        objaverse=dict(
            scale=0.90,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/kiwi"],
            scale=0.8,
        )
    ),
    knife=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.20,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/knife"],
            scale=1.0,
        )
    ),
    ladle=dict(
        types=("utensil"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(
            scale=1.10,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/ladle"],
            scale=1.2,
        )
    ),
    lemon=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/lemon"],
            scale=0.8,
        )
    ),
    lime=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        objaverse=dict(
            scale=1.0,
        ),
        aigen=dict(
            scale=0.90,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/lime"],
            scale=0.8,
        )
    ),
    mango=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.85,
            exclude=[
                "mango_3",  # one half is pitch dark
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/mango"],
            scale=0.8,
        )
    ),
    milk=dict(
        types=("dairy", "drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            exclude=[
                "milk_6"  # causing error: eigenvalues of mesh inertia violate A + B >= C
            ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/milk"],
            scale=1.0,
        )
    ),
    mug=dict(
        types=("receptacle", "stackable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.3,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/mug"],
            scale=0.9,
            exclude=["mug_52", "mug_97"], 
        )
    ),
    mushroom=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.20,
            exclude=[
                # "mushroom_16", # very very small holes. keeping anyway
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/mushroom"],
            scale=0.8,
            exclude=[
                "mushroom_2"
            ]
        )
    ),
    onion=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.1,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/onion"],
            scale=1.0,
        )
    ),
    orange=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(
            exclude=[
                # "orange_11", # bottom half is dark. keeping anyway
            ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/orange"],
            scale=1.0,
        )
    ),
    pan=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.25,
        ),
        objaverse=dict(
            scale=1.70,
            exclude=[
                "pan_16",  # causing error. faces of mesh have inconsistent orientation,
                "pan_0",
                "pan_12",
                "pan_17",
                "pan_22",  # these are technically what we consider "pots"
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pan"],
            scale=2.0,
            exclude=["pan_11", "pan_5", "pan_6"]
        )
    ),
    pot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=2.25,
        ),
        objaverse=dict(
            model_folders=["objaverse/pan"],
            scale=1.70,
            exclude=list(
                set([f"pan_{i}" for i in range(25)])
                - set(["pan_0", "pan_12", "pan_17", "pan_22"])
            ),
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pot"],
            scale=2.0,
        )
    ),
    peach=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.05,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/peach"],
            scale=0.8,
        )
    ),
    pear=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(),
        objaverse=dict(
            exclude=[
                "pear_4",  # has big hole. excluding
            ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pear"],
            scale=0.8,
        )
    ),
    plate=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.65,
        ),
        objaverse=dict(
            scale=1.35,
            exclude=[
                "plate_6",  # causing error: faces of mesh have inconsistent orientation.
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/plate"],
            scale=1.0,
            exclude=["plate_12", "plate_14", "plate_18"]
        )
    ),
    potato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.10,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/potato"],
            scale=0.7,
        )
    ),
    rolling_pin=dict(
        types=("tool"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.6,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                # "rolling_pin_5", # can see thru side handle edges, keeping anyway
                # "rolling_pin_1", # can see thru side handle edges, keeping anyway
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/rolling_pin"],
            scale=2.0,
        )
    ),
    scissors=dict(
        types=("tool"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.35,
        ),
        objaverse=dict(
            scale=1.15,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/scissors"],
            scale=1.0,
        )
    ),
    shaker=dict(
        types=("condiment"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
    ),
    soap_dispenser=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.7,
        ),
        objaverse=dict(
            exclude=[
                # "soap_dispenser_4", # can see thru body but that's fine if this is glass
            ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/soap_dispenser"],
            scale=1.0,
        )
    ),
    spatula=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.30,
        ),
        objaverse=dict(
            scale=1.10,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/spatula"],
            scale=1.0,
        )
    ),
    sponge=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.20,
        ),
        objaverse=dict(
            scale=0.90,
            # exclude=[
            #     "sponge_7", "sponge_1", # self turning due to single collision geom
            # ]
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/sponge"],
            scale=0.8,
        )
    ),
    spoon=dict(
        types=("utensil"),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.5,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/spoon"],
            scale=1.0,
        )
    ),
    spray=dict(
        types=("cleaner"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            scale=1.75,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/spray"],
            scale=1.0,
        )
    ),
    squash=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=1.15,
        ),
        objaverse=dict(
            exclude=[
                "squash_10",  # hole at bottom
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/squash"],
            scale=1.0,
        )
    ),
    steak=dict(
        types=("meat"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(
            scale=[1.0, 1.0, 2.0],
        ),
        objaverse=dict(
            scale=[1.0, 1.0, 2.0],
            exclude=[
                "steak_13",  # bottom texture completely messed up
                "steak_1",  # bottom texture completely messed up
                # "steak_9", # bottom with some minor issues, keeping anyway
            ],
        ),
    ),
    sweet_potato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        aigen=dict(),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/sweet_potato"],
            scale=1.0,
        )
    ),
    tangerine=dict(
        types=("fruit"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/tangerine"],
            scale=0.8,
        )
    ),
    teapot=dict(
        types=("receptacle"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(
            scale=1.25,
            exclude=[
                "teapot_9",  # hole on bottom
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/teapot"],
            scale=1.5,
        )
    ),
    tomato=dict(
        types=("vegetable"),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        aigen=dict(
            scale=1.25,
        ),
        objaverse=dict(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/tomato"],
            scale=0.8,
        )
    ),
    tray=dict(
        types=("receptacle"),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(scale=2.0),
        objaverse=dict(
            scale=1.80,
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/tray"],
            scale=2.0,
        )
    ),
    waffle=dict(
        types=("sweets"),
        graspable=False,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.75,
        ),
        objaverse=dict(
            exclude=[
                "waffle_2",  # bottom completely messed up
            ]
        ),
    ),
    water_bottle=dict(
        types=("drink"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.6,
        ),
        objaverse=dict(
            scale=1.5,
            exclude=[
                "water_bottle_11",  # sides and bottom see thru, but ok if glass. keeping anyway
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/water_bottle"],
            scale=1.5,
        )
    ),
    wine=dict(
        types=("drink", "alcohol"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        aigen=dict(
            scale=1.9,
        ),
        objaverse=dict(
            scale=1.6,
            exclude=[
                "wine_7",  # causing error. faces of mesh have inconsistent orientation
            ],
        ),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/wine"],
            scale=1.5,
        )
    ),
    yogurt=dict(
        types=("dairy", "packaged_food"),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        aigen=dict(
            scale=1.0,
        ),
        objaverse=dict(
            scale=0.95,
        ),
    ),
    dates=dict(
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
        aigen=dict(),
    ),
    lemonade=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("drink"),
    ),
    walnut=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=(),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/walnut"],
            scale=0.5,
        )
    ),
    cheese_grater=dict(
        aigen=dict(
            scale=2.15,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    syrup_bottle=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("condiment"),
    ),
    scallops=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    candy=dict(
        aigen=dict(),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("sweets"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/candy"],
            scale=0.7,
        )
    ),
    whisk=dict(
        aigen=dict(
            scale=1.8,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("utensil"),
    ),
    pitcher=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("receptacle"),
    ),
    ice_cream=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("sweets"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/ice_cream"],
            scale=1.0,
        )
    ),
    cherry=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    peanut_butter=dict(
        aigen=dict(
            scale=1.25,
            model_folders=["aigen_objs/peanut_butter_jar"],
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    thermos=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("drink"),
    ),
    ham=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    dumpling=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat", "cooked_food"),
    ),
    cabbage=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lettuce=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("vegetable"),
    ),
    tongs=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/tongs"],
            scale=1.5,
        )
    ),
    ginger=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    ice_cube_tray=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("receptacle"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/ice_cube_tray"],
            scale=1.5,
        )
    ),
    shrimp=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    cantaloupe=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cantaloupe"],
            scale=1.0,
        )
    ),
    honey_bottle=dict(
        aigen=dict(
            scale=1.10,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    grapes=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    spaghetti_box=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    chili_pepper=dict(
        aigen=dict(
            scale=1.10,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    celery=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    burrito=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("cooked_food"),
    ),
    olive_oil_bottle=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    kebabs=dict(
        aigen=dict(
            scale=1.65,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("cooked_food"),
    ),
    bottle_opener=dict(
        aigen=dict(),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("tool"),
    ),
    chicken_breast=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    jello_cup=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("packaged_food"),
    ),
    lobster=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    brussel_sprout=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    sushi=dict(
        aigen=dict(
            scale=0.90,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("meat"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/sushi"],
            scale=1.0,
        )
    ),
    baking_sheet=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("receptacle"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/baking_sheet"],
            scale=1.0,
        )
    ),
    wine_glass=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=False,
        freezable=True,
        types=("receptacle"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/wine_glass"],
            scale=1.0,
        )
    ),
    asparagus=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    lamb_chop=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    pickle=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("vegetable"),
    ),
    bacon=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=False,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=False,
        types=("meat"),
    ),
    canola_oil=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    strawberry=dict(
        aigen=dict(
            scale=0.9,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    watermelon=dict(
        aigen=dict(
            scale=2.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/watermelon"],
            scale=2.0,
            exclude=["watermelon_1"]
        )
    ),
    pizza_cutter=dict(
        aigen=dict(
            scale=1.4,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pizza"],
            scale=1.4,
        )
    ),
    pomegranate=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pomegranate"],
            scale=0.8,
        )
    ),
    apricot=dict(
        aigen=dict(
            scale=0.7,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
    ),
    beet=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        types=("vegetable"),
    ),
    radish=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("vegetable"),
    ),
    salsa=dict(
        aigen=dict(
            scale=1.15,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food"),
    ),
    artichoke=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=False,
        types=("vegetable"),
    ),
    scone=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("pastry", "bread_food"),
    ),
    hamburger=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("cooked_food"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/hamburger"],
            scale=1.0,
        )
    ),
    raspberry=dict(
        aigen=dict(
            scale=0.85,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
    ),
    tacos=dict(
        aigen=dict(
            scale=1.0,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=False,
        freezable=False,
        types=("cooked_food"),
    ),
    vinegar=dict(
        aigen=dict(
            scale=1.4,
        ),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("packaged_food", "condiment"),
    ),
    zucchini=dict(
        aigen=dict(
            scale=1.35,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
    ),
    pork_loin=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    pork_chop=dict(
        aigen=dict(
            scale=1.25,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pork_chop"],
            scale=1.0,
        )
    ),
    sausage=dict(
        aigen=dict(
            scale=1.45,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    coconut=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("fruit"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/coconut"],
            scale=1.0,
        )
    ),
    cauliflower=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("vegetable"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cauliflower"],
            scale=1.0,
        )
    ),
    lollipop=dict(
        aigen=dict(),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("sweets"),
    ),
    salami=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("meat"),
    ),
    butter_stick=dict(
        aigen=dict(
            scale=1.3,
        ),
        graspable=True,
        washable=False,
        microwavable=True,
        cookable=True,
        freezable=True,
        types=("dairy"),
    ),
    can_opener=dict(
        aigen=dict(
            scale=1.5,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=False,
        types=("tool"),
    ),
    tofu=dict(
        aigen=dict(),
        graspable=True,
        washable=True,
        microwavable=False,
        cookable=True,
        freezable=True,
        types=(),
    ),
    pineapple=dict(
        aigen=dict(
            scale=2.0,
        ),
        graspable=False,
        washable=True,
        microwavable=False,
        cookable=False,
        freezable=True,
        types=("fruit"),
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pineapple"],
            scale=1.5,
        )
    ),
    skewers=dict(
        aigen=dict(
            scale=1.75,
        ),
        graspable=True,
        washable=True,
        microwavable=True,
        cookable=True,
        freezable=False,
        types=("meat", "cooked_food"),
    ),
    barrel=dict(
        types=("receptable"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/barrel"],
            scale=2.0
        )
    ),
    bottle=dict(
        types=("drink"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/bottle"],
            scale=1.0
        ),
    ),
    coaster=dict(
        types=("decoration"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/coaster"],
            scale=1.0
        ),
    ),
    coffee_machine=dict(
        types=("tool"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/coffee_machine"],
            scale=1.2
        ),
    ),
    cookie=dict(
        types=("food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/cookie"],
            scale=0.7
        ),
    ),
    dessert=dict(
        types=("food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/dessert"],
            scale=1.0
        ),
    ),
    fruit=dict(
        types=("food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/fruit"],
            scale=1.0
        ),
    ),
    gadgett=dict(
        types=("tool"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/gadget"],
            scale=1.0
        ),
    ),
    glass=dict(
        types=("receptable"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/glass"],
            scale=1.0
        ),
    ),
    jar=dict(
        types=("receptable"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/jar"],
            scale=1.0
        ),
    ),
    melon=dict(
        types=("food"),
        graspable=True,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/melon"],
            scale=1.2
        ),
    ),
    peanut=dict(
        types=("food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/peanut"],
            scale=0.5
        ),
    ),
    pumpkin=dict(
        types=("food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/pumpkin"],
            scale=1.2
        ),
    ),
    sandwich=dict(
        types=("food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=True,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/sandwich"],
            scale=1.0
        ),
    ),
    tissue_box=dict(
        types=("food"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/tissue_box"],
            scale=1.8
        ),
    ),
    toaster=dict(
        types=("tool"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/toaster"],
            scale=1.8
        ),
    ),
    utensil=dict(
        types=("tool"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/utensil"],
            scale=1.0
        ),
    ),
    vase=dict(
        types=("decoration"),
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        objaverse_extra=dict(
            model_folders=["objaverse_extra/vase"],
            scale=1.0
        ),
    ),
)


def get_cats_by_type(types):
    """
    Retrieves a list of item keys from the global `OBJ_CATEGORIES` dictionary based on the specified types.

    Args:
        types (list): A list of valid types to filter items by. Only items with a matching type will be included.

    Returns:
        list: A list of keys from `OBJ_CATEGORIES` where the item's types intersect with the provided `types`.
    """
    types = set(types)

    res = []
    for key, val in OBJ_CATEGORIES.items():
        cat_types = val["types"]
        if isinstance(cat_types, str):
            cat_types = [cat_types]
        cat_types = set(cat_types)
        # Access the "types" key in the dictionary using the correct syntax
        if len(cat_types.intersection(types)) > 0:
            res.append(key)

    return res


### define all object categories ###
OBJ_GROUPS = dict(
    all=list(OBJ_CATEGORIES.keys()),
)

for k in OBJ_CATEGORIES:
    OBJ_GROUPS[k] = [k]

all_types = set()
# populate all_types
for (cat, cat_meta_dict) in OBJ_CATEGORIES.items():
    # types are common to both so we only need to examine one
    cat_types = cat_meta_dict["types"]
    if isinstance(cat_types, str):
        cat_types = [cat_types]
    all_types = all_types.union(cat_types)

# populate OBJ_GROUPS which maps types to categories associated with the type
for t in all_types:
    OBJ_GROUPS[t] = get_cats_by_type(types=[t])

OBJ_GROUPS["food"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)
OBJ_GROUPS["in_container"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)

# custom groups
OBJ_GROUPS["container"] = ["plate"]  # , "bowl"]
OBJ_GROUPS["kettle"] = ["kettle_electric", "kettle_non_electric"]
OBJ_GROUPS["cookware"] = ["pan", "pot", "kettle_non_electric"]
OBJ_GROUPS["pots_and_pans"] = ["pan", "pot"]
OBJ_GROUPS["food_set1"] = [
    "apple",
    "baguette",
    "banana",
    "carrot",
    "cheese",
    "cucumber",
    "egg",
    "lemon",
    "orange",
    "potato",
]
OBJ_GROUPS["group1"] = ["apple", "carrot", "banana", "bowl", "can"]
OBJ_GROUPS["container_set2"] = ["plate", "bowl"]


class ObjCat:
    """
    Class that encapsulates data for an object category.

    Args:
        name (str): name of the object category

        types (tuple) or (str): type(s)/categories the object belongs to. Examples include meat, sweets, fruit, etc.

        model_folders (list): list of folders containing the MJCF models for the object category

        exclude (list): list of model names to exclude

        graspable (bool): whether the object is graspable

        washable (bool): whether the object is washable

        microwavable (bool): whether the object is microwavable

        cookable (bool): whether the object is cookable

        freezable (bool): whether the object is freezable

        scale (float): scale of the object meshes/geoms

        solimp (tuple): solimp values for the object meshes/geoms

        solref (tuple): solref values for the object meshes/geoms

        density (float): density of the object meshes/geoms

        friction (tuple): friction values for the object meshes/geoms

        priority: priority of the object

        aigen_cat (bool): True if the object is an AI-generated object otherwise its an objaverse object
    """

    def __init__(
        self,
        name,
        types,
        model_folders=None,
        exclude=None,
        graspable=False,
        washable=False,
        microwavable=False,
        cookable=False,
        freezable=False,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 2),
        density=100,
        friction=(0.95, 0.3, 0.1),
        priority=None,
        aigen_cat=False,
        source="objaverse"
    ):
        self.name = name
        if not isinstance(types, tuple):
            types = (types,)
        self.types = types

        self.aigen_cat = aigen_cat

        self.graspable = graspable
        self.washable = washable
        self.microwavable = microwavable
        self.cookable = cookable
        self.freezable = freezable

        self.scale = scale
        self.solimp = solimp
        self.solref = solref
        self.density = density
        friction = (1, 1, 1) # !!!
        self.friction = friction
        self.priority = priority
        self.exclude = exclude or []

        if model_folders is None:
            # subf = "aigen_objs" if self.aigen_cat else "objaverse"
            # model_folders = ["{}/{}".format(subf, name)]
            subf = source
            model_folders = ["{}/{}".format(subf, name)]
        cat_mjcf_paths = []
        for folder in model_folders:
            cat_path = os.path.join(BASE_ASSET_ZOO_PATH, folder)
            for root, _, files in os.walk(cat_path):
                if "model.xml" in files:
                    model_name = os.path.basename(root)
                    if model_name in self.exclude:
                        continue
                    cat_mjcf_paths.append(os.path.join(root, "model.xml"))
        self.mjcf_paths = sorted(cat_mjcf_paths)

    def get_mjcf_kwargs(self):
        """
        returns relevant data to apply to the MJCF model for the object category
        """
        return deepcopy(
            dict(
                scale=self.scale,
                solimp=self.solimp,
                solref=self.solref,
                density=self.density,
                friction=self.friction,
                priority=self.priority,
            )
        )


# update OBJ_CATEGORIES with ObjCat instances. Maps name to the different registries it can belong to
# and then maps the registry to the ObjCat instance
for (name, kwargs) in OBJ_CATEGORIES.items():

    # get the properties that are common to both registries
    common_properties = deepcopy(kwargs)
    for k in common_properties.keys():
        assert k in [
            "graspable",
            "washable",
            "microwavable",
            "cookable",
            "freezable",
            "types",
            "aigen",
            "objaverse",
            "objaverse_extra"
        ]
    objaverse_kwargs = common_properties.pop("objaverse", None)
    aigen_kwargs = common_properties.pop("aigen", None)
    objaverse_extra_kwargs = common_properties.pop("objaverse_extra", None)
    assert "scale" not in kwargs
    OBJ_CATEGORIES[name] = {}

    # create instances
    if objaverse_kwargs is not None:
        objaverse_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["objaverse"] = ObjCat(name=name, source="objaverse", **objaverse_kwargs)
    if aigen_kwargs is not None:
        aigen_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["aigen"] = ObjCat(
            name=name, aigen_cat=True, source="aigen_objs", **aigen_kwargs
        )
    if objaverse_extra_kwargs is not None:
        objaverse_extra_kwargs.update(common_properties)
        OBJ_CATEGORIES[name]["objaverse_extra"] = ObjCat(name=name, source="objaverse_extra", **objaverse_extra_kwargs)



def sample_kitchen_object(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    freezable=None,
    rng=None,
    obj_registries=("objaverse", "objaverse_extra", "aigen"),
    split=None,
    max_size=(None, None, None),
    object_scale=None,
    cfg=None
):
    """
    Sample a kitchen object from the specified groups and within max_size bounds.

    Args:
        groups (list or str): groups to sample from or the exact xml path of the object to spawn

        exclude_groups (str or list): groups to exclude

        graspable (bool): whether the sampled object must be graspable

        washable (bool): whether the sampled object must be washable

        microwavable (bool): whether the sampled object must be microwavable

        cookable (bool): whether whether the sampled object must be cookable

        freezable (bool): whether whether the sampled object must be freezable

        rng (np.random.Generator): random number object

        obj_registries (tuple): registries to sample from

        split (str): split to sample from. Split "A" specifies all but the last 3 object instances
                    (or the first half - whichever is larger), "B" specifies the  rest, and None specifies all.

        max_size (tuple): max size of the object. If the sampled object is not within bounds of max size, function will resample

        object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


    Returns:
        dict: kwargs to apply to the MJCF model for the sampled object

        dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to, the category of the object
              the sampling split the object came from, and the groups the object was sampled from
    """
    # breakpoint()
    valid_object_sampled = False
    while valid_object_sampled is False:
        mjcf_kwargs, info = sample_kitchen_object_helper(
            groups=groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            rng=rng,
            obj_registries=obj_registries,
            split=split,
            object_scale=object_scale,
            cfg=cfg
        )

        # check if object size is within bounds
        mjcf_path = info["mjcf_path"]
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        # todo: bottom/top/horizaontal_radius
        bottom = string_to_array(
            find_elements(root=root, tags="site", attribs={"name": "bottom_site"}).get(
                "pos"
            )
        )
        top = string_to_array(
            find_elements(root=root, tags="site", attribs={"name": "top_site"}).get(
                "pos"
            )
        )
        horizontal_radius = string_to_array(
            find_elements(
                root=root, tags="site", attribs={"name": "horizontal_radius_site"}
            ).get("pos")
        )
        scale = mjcf_kwargs["scale"]
        obj_size = (
            np.array(
                [horizontal_radius[0] * 2, horizontal_radius[1] * 2, top[2] - bottom[2]]
            )
            * scale
        )
        valid_object_sampled = True
        for i in range(3):
            if max_size[i] is not None and obj_size[i] > max_size[i]:
                valid_object_sampled = False    

    print(info)
    return mjcf_kwargs, info


def sample_kitchen_object_helper(
    groups,
    exclude_groups=None,
    graspable=None,
    washable=None,
    microwavable=None,
    cookable=None,
    freezable=None,
    rng=None,
    obj_registries=("objaverse", "objaverse_extra", "aigen"),
    split=None,
    object_scale=None,
    cfg=None
):
        
    """
    Helper function to sample a kitchen object.

    Args:
        groups (list or str): groups to sample from or the exact xml path of the object to spawn

        exclude_groups (str or list): groups to exclude

        graspable (bool): whether the sampled object must be graspable

        washable (bool): whether the sampled object must be washable

        microwavable (bool): whether the sampled object must be microwavable

        cookable (bool): whether whether the sampled object must be cookable

        freezable (bool): whether whether the sampled object must be freezable

        rng (np.random.Generator): random number object

        obj_registries (tuple): registries to sample from

        split (str): split to sample from. Split "A" specifies all but the last 3 object instances
                    (or the first half - whichever is larger), "B" specifies the  rest, and None specifies all.

        object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


    Returns:
        dict: kwargs to apply to the MJCF model for the sampled object

        dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to, the category of the object
              the sampling split the object came from, and the groups the object was sampled from
    """
    if cfg and cfg.get('info', None):
        ori_info = cfg['info']
        cat = ori_info['cat']
        mjcf_path = ori_info['mjcf_path']
        mjcf_path = os.path.join(BASE_ASSET_ZOO_PATH, '/'.join(mjcf_path.split('/')[-4:]))
        chosen_reg = mjcf_path.split('/')[-4]
        mjcf_kwargs = OBJ_CATEGORIES[cat][chosen_reg].get_mjcf_kwargs()
        mjcf_kwargs['mjcf_path'] = mjcf_path
        ori_info['mjcf_path'] = mjcf_path
        if object_scale is not None:
            mjcf_kwargs['scale'] *= object_scale
        return mjcf_kwargs, ori_info
    
    if rng is None:
        rng = np.random.default_rng()

    # option to spawn specific object instead of sampling from a group
    if isinstance(groups, str) and groups.endswith(".xml"):
        mjcf_path = groups
        # reverse look up mjcf_path to category
        mjcf_kwargs = dict()
        cat = None
        obj_found = False
        for cand_cat in OBJ_CATEGORIES:
            for reg in obj_registries:
                if (
                    reg in OBJ_CATEGORIES[cand_cat]
                    and mjcf_path in OBJ_CATEGORIES[cand_cat][reg].mjcf_paths
                ):
                    mjcf_kwargs = OBJ_CATEGORIES[cand_cat][reg].get_mjcf_kwargs()
                    cat = cand_cat
                    obj_found = True
                    break
            if obj_found:
                break
        if obj_found is False:
            raise ValueError
        mjcf_kwargs["mjcf_path"] = mjcf_path
    else:
        if not isinstance(groups, tuple) and not isinstance(groups, list):
            groups = [groups]

        if exclude_groups is None:
            exclude_groups = []
        if not isinstance(exclude_groups, tuple) and not isinstance(
            exclude_groups, list
        ):
            exclude_groups = [exclude_groups]

        invalid_categories = []
        for g in exclude_groups:
            for cat in OBJ_GROUPS[g]:
                invalid_categories.append(cat)

        valid_categories = []
        for g in groups:
            for cat in OBJ_GROUPS[g]:
                # don't repeat if already added
                if cat in valid_categories:
                    continue
                if cat in invalid_categories:
                    continue

                # don't include if category not represented in any registry
                cat_in_any_reg = np.any(
                    [reg in OBJ_CATEGORIES[cat] for reg in obj_registries]
                )
                if not cat_in_any_reg:
                    continue

                invalid = False
                for reg in obj_registries:
                    if reg not in OBJ_CATEGORIES[cat]:
                        continue
                    cat_meta = OBJ_CATEGORIES[cat][reg]
                    if graspable is True and cat_meta.graspable is not True:
                        invalid = True
                    if washable is True and cat_meta.washable is not True:
                        invalid = True
                    if microwavable is True and cat_meta.microwavable is not True:
                        invalid = True
                    if cookable is True and cat_meta.cookable is not True:
                        invalid = True
                    if freezable is True and cat_meta.freezable is not True:
                        invalid = True

                if invalid:
                    continue

                valid_categories.append(cat)

        cat = rng.choice(valid_categories)
                
        
        choices = {reg: [] for reg in obj_registries}
        

        if cfg and cfg.get('target_obj_name', None) and "distr" in cfg["name"]:
            target_obj_name = cfg['target_obj_name']
            unique_attr = cfg['unique_attr']
            for reg in obj_registries:
                # breakpoint()
                if reg not in OBJ_CATEGORIES[cat]:
                    choices[reg] = []
                    continue
                tmp_choices = []
                for cate in valid_categories:
                    if reg not in OBJ_CATEGORIES[cate]:
                        continue
                    reg_choices = deepcopy(OBJ_CATEGORIES[cate][reg].mjcf_paths)
                    if split is not None:
                        split_th = max(len(choices) - 3, int(math.ceil(len(reg_choices) / 2)))
                        if split == "A":
                            reg_choices = reg_choices[:split_th]
                        elif split == "B":
                            reg_choices = reg_choices[split_th:]
                        else:
                            raise ValueError
                    tmp_choices.extend(reg_choices)
                    # breakpoint()
                choice_map = {}
                for path in tmp_choices:
                    source = path.split('/')[-4]
                    id = path.split('/')[-2]
                    obj_name = f"{source}_{id}"
                    # choice_map[path.split('/')[-2]] = path
                    choice_map[obj_name] = path
                # breakpoint()
                target_obj_info = ALL_OBJ_INFOS['obj_infos'][target_obj_name]
                unique_attr2objs = ALL_OBJ_INFOS[f"{unique_attr}2objs"]
                unique_attrs = target_obj_info[unique_attr]
                if type(unique_attrs) != list:
                    unique_attrs = [unique_attrs]
                for tmp_attr in unique_attrs:
                    for obj_name in unique_attr2objs[tmp_attr]:
                        if obj_name in choice_map:
                            del choice_map[obj_name]
                
                # ORI_RANK (n_objs, n_objs, 5)
                target_obj_ori_rank = ORI_RANK[CLASSNAME2IDX[target_obj_name]]
                valid_obj_idx_list = []
                for obj_name in choice_map.keys():
                    valid_obj_idx_list.append(CLASSNAME2IDX[obj_name])
                # breakpoint()
                
                # target_obj_sim = OBJ_SIM_MATRIX[CLASSNAME2IDX[target_obj_name]]
                # target_obj_sim[:, ATTR2IDX[unique_attr]] = 1 - target_obj_sim[:, ATTR2IDX[unique_attr]]
                # OBJ_SIM_MATRIX
                
                # attr_list = ['class', 'color', 'shape', 'material']
                # attr_list.remove(unique_attr)
                # remained_attr_idx_list = [ATTR2IDX[attr] for attr in attr_list]
                # tmp_rank = target_obj_ori_rank[valid_obj_idx_list][:, remained_attr_idx_list]
                
                tmp_rank = target_obj_ori_rank[valid_obj_idx_list].to(torch.float32)
                tmp_rank[:, ATTR2IDX[unique_attr]] *= -5
                # tmp_rank[:, ATTR2IDX[unique_attr]].clamp_min(-50)
                
                tmp_sum_rank = tmp_rank.sum(dim=-1)
                tmp_rank_idx = tmp_sum_rank.argsort(dim=-1)
                tmp_choices = [OBJ_NAME_LIST[valid_obj_idx_list[x]] for x in tmp_rank_idx[:30]]
                print(f"{reg}: {len(valid_obj_idx_list)}")
                choices[reg] = list(map(lambda x: choice_map[x], tmp_choices))
                # ct = defaultdict(int)
                # for attr in attr_list:
                #     tmp_attrs = target_obj_info[attr]
                #     if type(tmp_attrs) != list:
                #         tmp_attrs = [tmp_attrs]
                #     attr2objs = ALL_OBJ_INFOS[f"{attr}2objs"]
                #     for tmp_attr in tmp_attrs:
                #         for obj_name in attr2objs[tmp_attr]:
                #             if obj_name in choice_map:
                #                 ct[obj_name] += 1
                # tmp_choices = sorted(ct.items(), key = lambda item: item[1])[-10:]
                # choices[reg] = list(map(lambda x: choice_map[x[0]], tmp_choices))
        else:
            for reg in obj_registries:
                if reg not in OBJ_CATEGORIES[cat]:
                    choices[reg] = []
                    continue
                reg_choices = deepcopy(OBJ_CATEGORIES[cat][reg].mjcf_paths)
                if split is not None:
                    split_th = max(len(choices) - 3, int(math.ceil(len(reg_choices) / 2)))
                    if split == "A":
                        reg_choices = reg_choices[:split_th]
                    elif split == "B":
                        reg_choices = reg_choices[split_th:]
                    else:
                        raise ValueError
                choices[reg] = reg_choices
        
        chosen_reg = rng.choice(
            obj_registries,
            p=np.array([len(choices[reg]) for reg in obj_registries])
            / sum(len(choices[reg]) for reg in obj_registries),
        )
            
        mjcf_path = rng.choice(choices[chosen_reg])
        mjcf_kwargs = OBJ_CATEGORIES[cat][chosen_reg].get_mjcf_kwargs()
        mjcf_kwargs["mjcf_path"] = mjcf_path

    if object_scale is not None:
        mjcf_kwargs["scale"] *= object_scale
        
    groups_containing_sampled_obj = []
    for group, group_cats in OBJ_GROUPS.items():
        if cat in group_cats:
            groups_containing_sampled_obj.append(group)

    info = {
        "groups_containing_sampled_obj": groups_containing_sampled_obj,
        "groups": groups,
        "cat": cat,
        "split": split,
        "mjcf_path": mjcf_path,
    }

    # print(mjcf_path)

    return mjcf_kwargs, info
