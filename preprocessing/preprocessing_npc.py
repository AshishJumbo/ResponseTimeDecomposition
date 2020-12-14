import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import stats
# from sklearn.mixture import GaussianMixture
#
hint_action = pd.read_csv("../data/hint_action_GMM.csv")
all_hint_action = pd.read_csv("../data/all_hint_action_GMM.csv")
hint_hint_npc = pd.read_csv("../data/assignment_problem_npc_infos_with_priors.csv")
# hint_hint_npc = pd.read_csv("../data/assignment_problem_npc_infos.csv")

# hint_hint_npc['pr'] = (hint_hint_npc.path_info.astype(str) + "/").str.extract("\/PR(.*?)\#")
# hint_hint_npc["pr"] = hint_hint_npc["pr"].astype(int)
# hint_hint_npc.to_csv("../data/assignment_problem_npc_infos.csv", index=False)

hint_action = hint_action.merge(hint_hint_npc, on=["assignment_log_id", "pr", "user_xid"], how="left")
all_hint_action = all_hint_action.merge(hint_hint_npc, on=["assignment_log_id", "pr", "user_xid"], how="left")


hint_action.to_csv("../data/hint_action_GMM.csv", index=False)
all_hint_action.to_csv("../data/all_hint_action_GMM.csv", index=False)

# hint_hint_npc = pd.read_csv("../data/assignment_problem_npc_infos_with_priors.csv")
#
# hint_hint_npc['pr'] = (hint_hint_npc.path_info.astype(str) + "/").str.extract("\/PR(.*?)\#")
# hint_hint_npc["pr"] = hint_hint_npc["pr"].astype(int)
# hint_hint_npc.to_csv("../data/assignment_problem_npc_infos_with_priors.csv", index=False)