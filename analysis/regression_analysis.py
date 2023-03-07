import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

hint_action_GMM_npc = pd.read_csv("../data/hint_action_GMM_npc.csv")
all_hint_action_GMM_npc = pd.read_csv("../data/all_hint_action_GMM_npc.csv")

hint_action_GMM_npc['cluster'] = hint_action_GMM_npc['cluster'].map({1: 'low-efort', 0: 'high-effort'})
all_hint_action_GMM_npc['cluster'] = all_hint_action_GMM_npc['cluster'].map({1: 'low-efort', 0: 'high-effort'})

hint_action_GMM_npc["prior_percent_correct"].fillna((hint_action_GMM_npc["prior_percent_correct"].mean()), inplace=True)
all_hint_action_GMM_npc["prior_percent_correct"].fillna((all_hint_action_GMM_npc["prior_percent_correct"].mean()),
                                                        inplace=True)

hint_action_GMM_npc["prior_percent_correct"].replace(0, 0.00001, inplace=True)
all_hint_action_GMM_npc["prior_percent_correct"].replace(0, 0.00001, inplace=True)
hint_action_GMM_npc["prior_completion"].fillna((hint_action_GMM_npc["prior_completion"].mean()), inplace=True)
all_hint_action_GMM_npc["prior_completion"].fillna((all_hint_action_GMM_npc["prior_completion"].mean()),
                                                   inplace=True)

sns.set(context="poster", style="whitegrid")
sns.set(rc={'figure.figsize': (10, 8)})


def generateCorr(df):
    df = df[["log_action_action_pairs_time_taken", "cluster", "high_effort", "low_effort", "next_problem_correctness",
             "is_skill_builder", "assignment_wheel_spin", "completed", "prior_completion",
             "prior_percent_correct", "high_effort_area", "low_effort_area"]].copy()
    sns.heatmap(df.corr(), annot=True)
    plt.show()


generateCorr(hint_action_GMM_npc)
generateCorr(all_hint_action_GMM_npc)

hint_action_GMM_npc["high_low"] = 0
hint_action_GMM_npc.loc[hint_action_GMM_npc["high_effort_area"] >= 0.5, "high_low"] = 1

hint_action_GMM_npc["low_high"] = 0
hint_action_GMM_npc.loc[hint_action_GMM_npc["low_effort_area"] <= 0.5, "low_high"] = 1


def generateOLSModels(data):
    model1 = logit("next_problem_correctness ~ high_low + low_high + prior_percent_correct",
                data=data)
    results = model1.fit()
    print(results.summary())

    model2 = logit("assignment_wheel_spin ~  high_low + low_high  + prior_completion",
                data=data)
    results = model2.fit()
    print(results.summary())

    model3 = logit("completed ~  high_low + low_high  + prior_completion",
                  data=data)
    results = model3.fit()
    print(results.summary())


print("====================================================================================================")
print("==============================================Hint_action===========================================")
print("====================================================================================================")
generateOLSModels(hint_action_GMM_npc)

# print("====================================================================================================")
# print("==========================================all_Hint_action===========================================")
# print("====================================================================================================")
# generateOLSModels(all_hint_action_GMM_npc)
#
#
