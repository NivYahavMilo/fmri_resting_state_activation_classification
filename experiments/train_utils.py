from enums import PreprocessType, Mode, FlowType
from experiments.preprocess_utils import preprocess_dataset


def get_supervised_tensors(roi: str, mode: Mode, preprocess_type: PreprocessType, rest_window: tuple, data_type: FlowType):
    dataset_name = f'{preprocess_type.value.lower()}_{data_type.value.lower()}_{roi}_roi_{rest_window}_rest_window'
    dataset = preprocess_dataset(
        roi=roi,
        dataset=f'{dataset_name}.pkl',
        mode=mode,
        preprocess_type=preprocess_type
    )
    # Remove columns that don't represent movie's features
    y = dataset['y'].tolist()
    X = dataset.drop(['subject', 'mode', 'y'], axis=1).values
    return X, y
