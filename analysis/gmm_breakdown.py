import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

hint_action = pd.read_csv("../data/hint_action_GMM.csv")
hint_hint = pd.read_csv("../data/hint_hint_GMM.csv")
all_hint_action = pd.read_csv("../data/all_hint_action_GMM.csv")
all_hint_hint = pd.read_csv("../data/all_hint_hint_GMM.csv")

sns.set(context="poster", style="whitegrid")
sns.set(rc={'figure.figsize': (16, 12)})
sns.distplot(hint_action.log_action_action_pairs_time_taken.values, hist=True, kde=True, rug=False,
             label="hint_action: " + str(len(hint_action)))
plt.legend()
plt.title("hint_action RTD of students who asked for the first hint digested it and asked for the next hint \n z-score("
          "-3, 3)")
plt.show()

clusters = 2

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

labels = gmm.predict(hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
labels2 = gmm.predict_proba(hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))

effort = pd.DataFrame(labels2, columns=list('xy'))

hint_action["cluster"] = labels.tolist()
hint_action["high_effort"] = effort['x']
hint_action["low_effort"] = effort['y']

test = hint_action[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                  "cluster", "high_effort", "low_effort", "next_problem_correctness", "is_skill_builder",
                  "assignment_wheel_spin", "completed"]].copy()

hint_action.to_csv("../data/hint_action_GMM_npc2.csv", index=False)

sns.set(context="poster", style="whitegrid")
sns.set(rc={'figure.figsize': (16, 12)})
sns.distplot(hint_action.log_action_action_pairs_time_taken.values, hist=True, kde=True, rug=False,
             label="hint_action: " + str(len(hint_action)))
sns.distplot(all_hint_action.log_action_action_pairs_time_taken.values, hist=True, kde=True, rug=False,
             label="all_hint_action: " + str(len(all_hint_action)))
plt.legend()
plt.title("all_hint_action RTD of students who asked for the first hint digested it and asked for the next hint \n z-score("
          "-3, 3)")
plt.show()

clusters = 2

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    all_hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

labels = gmm.predict(all_hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
labels2 = gmm.predict_proba(all_hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))

effort = pd.DataFrame(labels2, columns=list('xy'))

all_hint_action["cluster"] = labels.tolist()
all_hint_action["high_effort"] = effort['x']
all_hint_action["low_effort"] = effort['y']

test = all_hint_action[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                  "cluster", "high_effort", "low_effort", "next_problem_correctness", "is_skill_builder",
                  "assignment_wheel_spin", "completed"]].copy()

all_hint_action.to_csv("../data/all_hint_action_GMM_npc2.csv", index=False)