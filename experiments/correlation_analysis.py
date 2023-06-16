import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from enums import PreprocessType, FlowType


class DecodingCorrelationAnalysis:

    def load_raw_data(self, roi, data_type: PreprocessType, flow_type: FlowType):
        raw_data_path = os.path.join(config.RAW_DATASETS_PATH, roi)
        data_type = f'{data_type.value.lower()}' \
                    f'_{flow_type.value.lower()}' \
                    f'_{roi}_roi_(13, 18)_rest_window.pkl'

        roi_data_path = os.path.join(raw_data_path, data_type)
        data_df = pd.read_pickle(roi_data_path)

        return data_df

    def correlate_group(self, group_data: dict, shuffle: bool):
        rest_df = group_data['rest']
        if shuffle:
            rest_df = rest_df.sample(frac=1).reset_index(drop=True)

        movie_df = group_data['task']

        rest_df_name = movie_df.columns.tolist()

        correlation_matrix = np.zeros((14, 14))

        for i in range(14):
            for j in range(14):
                correlation_matrix[i, j] = np.corrcoef(rest_df.values[:, i], movie_df.values[:, j])[0, 1]

        correlation_matrix_df = pd.DataFrame(correlation_matrix, index=rest_df_name, columns=rest_df_name)

        return correlation_matrix_df

    def plot_and_save(self, df_result: pd.DataFrame, title, roi: str, preprocess: PreprocessType, shuffle: bool):

        shuffle = 'shuffled' if shuffle else ''
        title = f"{shuffle.title()} Group {title} Movies-Rest Correlation Heatmap".strip()

        save_fig_dir = os.path.join(config.FIGURES_PATH, roi, preprocess.value, shuffle)
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)

        save_fig_path = os.path.join(save_fig_dir, f'{title}.png')
        if os.path.isfile(save_fig_path):
            return

        plt.title(title)
        # Plot the heatmap using df.plot with kind='imshow'
        plt.imshow(df_result, cmap='coolwarm', interpolation='nearest')
        # Set the x-axis tick labels
        plt.xticks(ticks=np.arange(len(df_result.columns)), labels=df_result.columns)
        plt.yticks(ticks=np.arange(len(df_result.columns)), labels=df_result.columns)

        # Rotate x-axis tick labels for better visibility
        plt.xticks(rotation=45)

        colorbar = plt.colorbar()

        # Set the label for the colorbar
        colorbar.set_label('Correlation')
        plt.savefig(save_fig_path, dpi=300)
        plt.show()

    def flow(self, *args, **kwargs):
        shuffle = kwargs.pop('shuffle_rest', False)
        data = self.load_raw_data(**kwargs)

        group_score_average = []
        for group_index, group_data in data.items():
            group_score_matrix = self.correlate_group(group_data=group_data, shuffle=shuffle)
            self.plot_and_save(df_result=group_score_matrix, title=group_index, roi=kwargs['roi'],
                               preprocess=kwargs['data_type'], shuffle=shuffle)

            group_score_average.append(group_score_matrix)

        # Concatenate the DataFrames in the list vertically
        concatenated_df = pd.concat(group_score_average)

        # Group by index or any other column(s) if applicable
        grouped_df = concatenated_df.groupby(concatenated_df.index)

        # Calculate the mean of each group
        averaged_df = grouped_df.mean()

        self.plot_and_save(df_result=averaged_df, title='Averaged', roi=kwargs['roi'], preprocess=kwargs['data_type'],
                           shuffle=shuffle)


if __name__ == '__main__':
    decoding_analysis = DecodingCorrelationAnalysis()
    decoding_analysis.flow(
        roi='RH_Default_pCunPCC_1',
        data_type=PreprocessType.ACTIVATIONS,
        flow_type=FlowType.GROUP_SUBJECTS,
        shuffle_rest=True
    )
