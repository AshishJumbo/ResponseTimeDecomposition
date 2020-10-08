import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

hint_hint = pd.read_csv("../data/hint_hint_GMM.csv")
hint_hint_npc = pd.read_csv("../data/assignment_problem_npc_infos.csv")

hint_hint = hint_hint.loc[:, ~hint_hint.columns.str.contains('^Unnamed')]

sns.set(context="poster", style="whitegrid")
sns.set(rc={'figure.figsize': (16, 12)})
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
plt.legend()
plt.title("hint_hint RTD of students who asked for the first hint digested it and asked for the next hint \n z-score("
          "-3, 3)")
plt.show()

clusters = 2

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

labels = gmm.predict(hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))
labels2 = gmm.predict_proba(hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))

effort = pd.DataFrame(labels2, columns=list('xy'))

hint_hint["cluster"] = labels.tolist()
hint_hint["high_effort"] = effort['x']
hint_hint["low_effort"] = effort['y']

test = hint_hint[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                  "cluster", "high_effort", "low_effort", "next_problem_correctness", "is_skill_builder",
                  "assignment_wheel_spin", "completed"]].copy()


all_hint_hint = pd.read_csv("../data/all_hint_hint_GMM.csv")

all_hint_hint = all_hint_hint.loc[:, ~all_hint_hint.columns.str.contains('^Unnamed')]

sns.distplot(all_hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint: " + str(len(all_hint_hint)))
plt.legend()
plt.title("all_hint_hint RTD of students who asked for the first hint digested it and asked for the next hint \n "
          "z-score(-3, 3)")
plt.show()

clusters = 2

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    all_hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

labels = gmm.predict(all_hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))
labels2 = gmm.predict_proba(all_hint_hint.log_action_action_pairs_time_taken.values.reshape(-1, 1))

effort = pd.DataFrame(labels2, columns=list('xy'))

all_hint_hint["cluster"] = labels.tolist()
all_hint_hint["high_effort"] = effort['x']
all_hint_hint["low_effort"] = effort['y']

all_test = all_hint_hint[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                          "cluster", "high_effort", "low_effort", "next_problem_correctness", "is_skill_builder",
                          "assignment_wheel_spin", "completed"]].copy()

hint_attempt = pd.read_csv("../data/hint_attempt_GMM.csv")

hint_attempt = hint_attempt.loc[:, ~hint_attempt.columns.str.contains('^Unnamed')]

sns.distplot(hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_attempt: " + str(len(hint_attempt)))
plt.legend()
plt.title("hint_attempt RTD of students who asked for the first hint digested it and asked for the next hint \n "
          "z-score(-3, 3)")
plt.show()

clusters = 1

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    hint_attempt.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

all_test_h_a = hint_attempt[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                          "next_problem_correctness", "is_skill_builder",
                          "assignment_wheel_spin", "completed"]].copy()

all_hint_attempt = pd.read_csv("../data/all_hint_attempt_GMM.csv")

all_hint_attempt = all_hint_attempt.loc[:, ~all_hint_attempt.columns.str.contains('^Unnamed')]

sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt: " + str(len(all_hint_attempt)))
plt.legend()
plt.title("all_hint_attempt RTD of students who asked for the first hint digested it and asked for the next hint \n "
          "z-score(-3, 3)")
plt.show()

clusters = 1

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    all_hint_attempt.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

all_test_h_a = all_hint_attempt[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                          "next_problem_correctness", "is_skill_builder",
                          "assignment_wheel_spin", "completed"]].copy()
