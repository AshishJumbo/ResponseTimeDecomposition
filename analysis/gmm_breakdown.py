import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
import scipy

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
    hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

labels = gmm.predict(hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
labels2 = gmm.predict_proba(hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))

effort = pd.DataFrame(labels2, columns=list('xy'))

hint_action["cluster"] = labels.tolist()
hint_action["high_effort"] = effort['x']
hint_action["low_effort"] = effort['y']

means = [gmm.means_[0][0], gmm.means_[1][0]]
sds = np.sqrt(gmm.covariances_)
sigma = [sds[0][0][0], sds[1][0][0]]

hint_action["high_effort_area"] = scipy.stats.norm.cdf(((hint_action["log_action_action_pairs_time_taken"] - means[0]) / sigma[0]))
hint_action["low_effort_area"] = scipy.stats.norm.cdf(((hint_action["log_action_action_pairs_time_taken"] - means[1]) / sigma[1]))
hint_action.loc[hint_action["high_effort_area"] < 0, "high_effort_area"] = 0
hint_action.loc[hint_action["high_effort_area"] > 100, "high_effort_area"] = 100
hint_action.loc[hint_action["low_effort_area"] < 0, "low_effort_area"] = 0
hint_action.loc[hint_action["low_effort_area"] > 100, "low_effort_area"] = 100

test = hint_action[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                  "cluster", "high_effort", "low_effort", "next_problem_correctness", "is_skill_builder",
                  "assignment_wheel_spin", "completed"]].copy()

# hint_action.to_csv("../data/hint_action_GMM_npc.csv", index=False)

sns.set(context="poster", style="whitegrid")
sns.set(rc={'figure.figsize': (16, 12)})
sns.distplot(hint_action.log_action_action_pairs_time_taken.values, hist=True, kde=True, rug=False,
             label="hint_action: " + str(len(hint_action)))
sns.distplot(all_hint_action.log_action_action_pairs_time_taken.values, hist=True, kde=True, rug=False,
             label="all_hint_action: " + str(len(all_hint_action)))
plt.legend()
plt.title(
    "all_hint_action RTD of students who asked for the first hint digested it and asked for the next hint \n z-score("
    "-3, 3)")
plt.show()

clusters = 2

gmm_all = GaussianMixture(n_components=clusters, max_iter=100).fit(
    all_hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm_all.means_)

labels_all = gmm_all.predict(all_hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
labels2_all = gmm_all.predict_proba(all_hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))

effort_all = pd.DataFrame(labels2_all, columns=list('xy'))

all_hint_action["cluster"] = labels_all.tolist()
all_hint_action["high_effort"] = effort_all['x']
all_hint_action["low_effort"] = effort_all['y']

means = [gmm_all.means_[0][0], gmm_all.means_[1][0]]
sds = np.sqrt(gmm_all.covariances_)
sigma = [sds[0][0][0], sds[1][0][0]]

# p_values = scipy.stats.norm.sf(abs(z_scores))*2 #twosided
# NOTE: this need to be modified
all_hint_action["high_effort_area"] = scipy.stats.norm.sf(abs((all_hint_action["log_action_action_pairs_time_taken"] - means[0]) / sigma[0]))*2
all_hint_action["low_effort_area"] = scipy.stats.norm.sf(abs((all_hint_action["log_action_action_pairs_time_taken"] - means[1]) / sigma[1]))*2

all_hint_action.loc[all_hint_action["high_effort_area"] < 0, "high_effort_area"] = 0
all_hint_action.loc[all_hint_action["high_effort_area"] > 100, "high_effort_area"] = 100
all_hint_action.loc[all_hint_action["low_effort_area"] < 0, "low_effort_area"] = 0
all_hint_action.loc[all_hint_action["low_effort_area"] > 100, "low_effort_area"] = 100
test = all_hint_action[["action_action_pairs", "action_action_pairs_time_taken", "log_action_action_pairs_time_taken",
                  "cluster", "high_effort", "low_effort", "next_problem_correctness", "is_skill_builder",
                  "assignment_wheel_spin", "completed"]].copy()

# all_hint_action.to_csv("../data/all_hint_action_GMM_npc.csv", index=False)

print("NOTE!! make sure the clusters are arranged in a descending order else the high/low effort will get reversed")
