from pathlib import Path
import sys
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

import os.path as osp
from datasets.custom import CustomDataset
from builder import DATASETS
from utils import get_attr,time_features
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np



@DATASETS.register_module()
class AAAI_mse_Dataset(CustomDataset):
    def __init__(self,**kwargs):
        super(AAAI_mse_Dataset, self).__init__()
        self.root_path = osp.join(ROOT, get_attr(kwargs, "data_path", None))

        self.flag = get_attr(kwargs, "flag", None)
        if self.flag =='train':
            self.data_path = osp.join(self.root_path, get_attr(kwargs, "train_path", None))
        elif self.flag =='valid':
            self.data_path = osp.join(self.root_path, get_attr(kwargs, "valid_path", None))
        else:
            self.data_path = osp.join(self.root_path, get_attr(kwargs, "test_path", None))

        self.tech_indicator_list = get_attr(kwargs, "tech_indicator_list", None)
        self.size = get_attr(kwargs, "size", None)
        self.features = get_attr(kwargs, "features", 'M')
        self.target = get_attr(kwargs, "target", None)
        self.scale = get_attr(kwargs, "scale", True)
        self.timeenc = get_attr(kwargs, "timeenc", 0)
        self.freq = get_attr(kwargs, "freq", 'H')

        # size [seq_len, label_len, pred_len]
        # info
        if self.size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = self.size[0]
            self.label_len = self.size[1]
            self.pred_len = self.size[2]


        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path,index_col=0)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # Create a MultiIndex with [index, 'tic']
        df_raw.set_index(['date', 'tic'], inplace=True)
        self.tics = df_raw.index.get_level_values(1).unique()
        self.num_tics = len(self.tics)


        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data = df_raw[self.tech_indicator_list]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        try:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = []
            seq_y = []
            return_data = []

            tech_indicator_list = [col for col in self.data.columns if col not in ['returns']]
            tic_data_first = self.data.xs(self.tics[0], level=1)  # 첫 번째 티커의 데이터
            if s_end >= len(tic_data_first) or s_end + 1 >= len(tic_data_first):
                raise IndexError(f"s_end index out of range: s_end={s_end}, len(tic_data_first)={len(tic_data_first)}")
            date_data = tic_data_first.index[s_end].strftime('%Y-%m-%d')


            for tic in self.tics:
                tic_data = self.data.xs(tic, level=1)
                if s_end >= len(tic_data) or s_end + 1 >= len(tic_data):
                    raise IndexError(
                        f"s_end + 1 index out of range for tic={tic}: s_end={s_end}, len(tic_data)={len(tic_data)}")
                seq_x.append(tic_data[s_begin:s_end][tech_indicator_list].values)
                seq_y.append(tic_data[r_begin:r_end][tech_indicator_list].values)

                # close 가격의 비율 계산
                ratio = tic_data.iloc[s_end]['label']
                return_data.append(ratio)



            seq_x = np.array(seq_x)
            seq_y = np.array(seq_y)
            return_data = np.array(return_data)

            seq_x_mark = self.data_stamp[s_begin * self.num_tics: s_end * self.num_tics].reshape(self.num_tics,-1,self.data_stamp.shape[-1])
            seq_y_mark = self.data_stamp[r_begin * self.num_tics: r_end * self.num_tics].reshape(self.num_tics,-1,self.data_stamp.shape[-1])


            return seq_x, seq_y, seq_x_mark, seq_y_mark,return_data,date_data

        except IndexError as e:
            print(f"IndexError at index {index}: {e}")
            raise


    def __len__(self):
        first_tic = self.tics[0]
        tic_data = self.data.xs(first_tic,level=1)
        return len(tic_data) - self.seq_len - self.pred_len -1
