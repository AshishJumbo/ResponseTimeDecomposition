import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

hint_hint = pd.read_csv("../data/hint_hint_GMM.csv")
all_hint_hint = pd.read_csv("../data/all_hint_hint_GMM.csv")
all_hint_attempt = pd.read_csv("../data/all_hint_attempt_GMM.csv")
hint_attempt = pd.read_csv("../data/hint_attempt_GMM.csv")

hint_hint_npc = pd.read_csv("../data/assignment_problem_npc_infos.csv")

# hint_hint_npc['pr'] = (hint_hint_npc.path_info.astype(str) + "/").str.extract("\/PR(.*?)\#")
# hint_hint_npc["pr"] = hint_hint_npc["pr"].astype(int)
# hint_hint_npc.to_csv("../data/assignment_problem_npc_infos.csv", index=False)

hint_hint = hint_hint.merge(hint_hint_npc, on=["assignment_log_id", "pr", "user_xid"], how="left")
all_hint_hint = all_hint_hint.merge(hint_hint_npc, on=["assignment_log_id", "pr", "user_xid"], how="left")
all_hint_attempt = all_hint_attempt.merge(hint_hint_npc, on=["assignment_log_id", "pr", "user_xid"], how="left")
hint_attempt = hint_attempt.merge(hint_hint_npc, on=["assignment_log_id", "pr", "user_xid"], how="left")

hint_hint.to_csv("../data/hint_hint_GMM.csv", index=False)
hint_attempt.to_csv("../data/hint_attempt_GMM.csv", index=False)
all_hint_hint.to_csv("../data/all_hint_hint_GMM.csv", index=False)
all_hint_attempt.to_csv("../data/all_hint_attempt_GMM.csv", index=False)
