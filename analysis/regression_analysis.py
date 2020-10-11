import pandas as pd
from statsmodels.formula.api import ols

hint_action_GMM_npc = pd.read_csv("../data/hint_action_GMM_npc.csv")
all_hint_action_GMM_npc = pd.read_csv("../data/all_hint_action_GMM_npc.csv")
hint_action_GMM_npc2 = pd.read_csv("../data/hint_action_GMM_npc2.csv")
all_hint_action_GMM_npc2 = pd.read_csv("../data/all_hint_action_GMM_npc2.csv")

# correlation showed poor correlation between features
# high effort to wheelspin is -0.29
# low effort to wheelspin is -0.29
# cluster (0 high, 1 low) to nps 0.05


def generateOLSModels(data):
    model = ols("next_problem_correctness ~ high_effort + low_effort", data=data)
    results = model.fit()
    print(results.summary())

    model = ols("assignment_wheel_spin ~ high_effort + low_effort", data=data)
    results = model.fit()
    print(results.summary())


generateOLSModels(hint_action_GMM_npc)
print("====================================================================================================")
print("==========================================all_Hint_action===========================================")
print("====================================================================================================")
generateOLSModels(all_hint_action_GMM_npc)

print("====================================================================================================")
print("===================================all_Hint_action_on_hint_hint=====================================")
print("====================================================================================================")
generateOLSModels(hint_action_GMM_npc2)

print("====================================================================================================")
print("===================================all_Hint_action_on_hint_hint=====================================")
print("====================================================================================================")
generateOLSModels(all_hint_action_GMM_npc2)
