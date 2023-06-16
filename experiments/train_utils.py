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
    dataset = dataset.drop(['mode'], axis=1)

    # Rename the column 'subject' to 'group'
    dataset = dataset.rename(columns={'subject': 'group'})
    return dataset
