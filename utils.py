def get_clip_index_mapping(inverse: bool = False):
    clip_mapper = {
        "0": "testretest",
        "1": "twomen",
        "2": "bridgeville",
        "3": "pockets",
        "4": "overcome",
        "5": "inception",
        "6": "socialnet",
        "7": "oceans",
        "8": "flower",
        "9": "hotel",
        "10": "garden",
        "11": "dreary",
        "12": "homealone",
        "13": "brokovich",
        "14": "starwars"
    }

    if inverse:
        clip_mapper = {v: k for k, v in clip_mapper.items()}

    return clip_mapper
