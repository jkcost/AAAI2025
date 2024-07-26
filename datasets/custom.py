from builder import DATASETS
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(self, **kwargs):
        super(CustomDataset, self).__init__()

    def __len__(self):
        """Total number of samples of data."""
        raise NotImplementedError

# @DATASETS.register_module()
# class CustomDataset(Dataset):
#     def __init__(self, data, window_len):
#         self.data = data.values if isinstance(data, pd.DataFrame) else data
#         self.window_len = window_len
#
#     def __getitem__(self, index):
#         # index는 DailyBatchSamplerRandom에 의해 배치의 시작 인덱스를 제공받음
#         # [sample_length, window_len, feature] 형태로 반환
#         start_idx = index
#         end_idx = start_idx + self.window_len
#         return self.data[start_idx:end_idx]
#
#     def __len__(self):
#         # 데이터셋의 전체 길이
#         return len(self.data) - self.window_len + 1
#
#     def get_index(self):
#         # 데이터셋의 인덱스를 반환
#         return self.data.index if isinstance(self.data, pd.DataFrame) else np.arange(len(self.data))