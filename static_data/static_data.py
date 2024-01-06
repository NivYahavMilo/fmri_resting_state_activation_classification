import json
import os
from typing import List, Optional, Dict

import config


class StaticData:
    ROI_NAMES: Optional[List]
    SUBJECTS: Optional[List]
    CLIPS_ORDER: Optional[List]
    REST_ORDER: Optional[List]
    CLIP_MAPPING: Optional[Dict]
    SCAN_MAPPING: Optional[Dict]

    @classmethod
    def inhabit_class_members(cls):
        """
        load json file
        set class attr
        """
        data_path = os.path.join(config.STATIC_DATA_PATH, 'static_data.json')
        io = open(data_path)
        data = json.load(io)

        for attr, values in data.items():
            setattr(cls, attr, values)
