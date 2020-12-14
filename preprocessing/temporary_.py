import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from bs4 import BeautifulSoup
import re

# NOTE: the same as the other file only instead of randomly
#  winzorising the data we are taking logs and z-scoring
#  them to have an easier time justifying things later on
df_main = pd.read_csv('../data/RTD_data_randomsample_20K_new.csv')
df_hint_info = pd.read_csv("../data/hint_infos.csv")

print(df_main.columns)
unique_users = df_main.user_xid.unique()
unique_ps = df_main.ps.unique()
unique_pr = df_main.pr.unique()

start_time = df_main.assignment_start
end_time = df_main.assignment_end
print(df_main.shape)
print(df_main.action_type.unique())
print(df_main.columns)
description = df_main.describe()

action_pair_time = df_main.action_action_pairs_time_taken.unique()

test0 = df_main.action_action_pairs.value_counts()

df_main = df_main[df_main.pr != -1]
df_main = df_main[df_main.action_action_pairs_time_taken >= 1]
test1 = df_main.action_action_pairs.value_counts()
hint_request_counts = df_main.pr_hints_requested_by_user.value_counts()

all_hint_hint_ = df_main[(df_main.action_action_pairs == "HintRequestedAction_HintRequestedAction") &
                         (df_main.action_action_pairs_order <= 4) & (df_main.pr_hints_requested_by_user == 2)]
all_hint_hint_ = all_hint_hint_[all_hint_hint_.action_action_pairs_time_taken > 1.5]

print(df_main.groupby(['user_xid', 'pr', 'pr_hints_requested_by_user']).size().reset_index().rename(columns={0:'count'}))