from enum import Enum


class Mode(Enum):
    RESTING_STATE_TASK = 'RESTING_STATE_TASK'
    RESTING_STATE_REST = 'RESTING_STATE_REST'
    REST = 'REST'
    TASK = 'TASK'
    FIRST_REST_SECTION = 'FIRST_REST_SECTION'


class PreprocessType(Enum):
    ACTIVATIONS = 'activations'
    DISTANCES = 'distances'


class FlowType(Enum):
    SINGLE_SUBJECT = 'SINGLE_SUBJECT'
    GROUP_SUBJECTS = 'GROUP_SUBJECTS'
