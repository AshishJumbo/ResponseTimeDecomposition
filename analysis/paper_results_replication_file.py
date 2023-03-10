import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from bs4 import BeautifulSoup
from sklearn.mixture import GaussianMixture
import scipy
import re
from statsmodels.formula.api import logit
pd.options.mode.chained_assignment = None  # default='warn'

# NOTE: the same as the other file only instead of randomly winzorising the data we are taking logs and z-scoring
#  them to have an easier time justifying things later on
df_main = pd.read_csv('../data/RTD_data_randomsample_20K_new.csv')
df_hint_info = pd.read_csv("../data/hint_infos.csv")


# this was done to explore the influence of systemic factors on RTD
# I have not included the code here in the replication section, but you can search for it in the
# ../analysis/exploratory_data_analysis_log_z.py
def hint_body_clean_script():
    hint_bodies = df_hint_info.hint_body
    hint_bodies_clean = []
    hint_bodies_clean2 = []
    for hint_body in hint_bodies:
        soup = BeautifulSoup(hint_body, features="html.parser")
        text = soup.get_text()
        text = text.replace("\n", " ")

        hint_bodies_clean.append(text)
        hint_body = re.sub('<[^<]+?>', '', hint_body)
        hint_body = hint_body.replace("\n", " ")
        hint_body = hint_body.replace(u'\xa0', " ")
        hint_body = hint_body.replace("&nbsp;", ' ')
        hint_body = ' '.join(hint_body.split())
        hint_bodies_clean2.append(hint_body)

    df_hint_info["hint_body_clean"] = hint_bodies_clean
    df_hint_info["hint_body_clean2"] = hint_bodies_clean2
    df_hint_info["hint_body_length"] = df_hint_info['hint_body_clean'].str.len()
    df_hint_info["hint_body_has_table"] = df_hint_info['hint_body'].str.contains("tbody")
    df_hint_info["hint_body_td_count"] = df_hint_info['hint_body'].str.contains("<td")
    df_hint_info["hint_body_wiris_count"] = df_hint_info['hint_body'].str.contains("Wirisformula")
    df_hint_info["hint_body_has_wiris"] = df_hint_info['hint_body'].str.contains("wiris")
    df_hint_info["hint_body_word_count"] = df_hint_info.hint_body_clean.str.split().str.len()
    df_hint_info["hint_body_word_count2"] = \
        df_hint_info.hint_body_clean2.str.split().str.len() + df_hint_info.hint_body_td_count + \
        df_hint_info.hint_body_wiris_count


hint_body_clean_script()

df_main.action_action_pairs_time_taken.replace({0: 0.000000001}, inplace=True)

# automated computer generated actions that don't convey much information are getting dropped
df_main = df_main[df_main.action_action_pairs != "ProblemFinishedAction_ProblemSetFinishedAction"]
# problemset level action so it's significance is already reflected by the
# StudentResponseAction_UserSelectedContinueAction
df_main = df_main[df_main.action_action_pairs != "UserSelectedContinueAction"]
# dropping all ps level actions because the pr level pairing already captures these.
df_main = df_main[df_main.pr != -1]

all_hint_action = df_main[df_main.action_action_pairs.str.contains("HintRequestedAction_")]
# print(all_hint_attempt.action_action_pairs.value_counts())
# 4 action paris would mean at least one hint request and at most 3 attempts
# ASSISTments penalizes students with a 30% penalty per help request or incorrect attempt
# hence by the 4th action the learners ability to earn credit on the problem becomes 0
all_hint_attempt = all_hint_action[
    (all_hint_action.action_action_pairs == "HintRequestedAction_StudentResponseAction") &
    (all_hint_action.action_action_pairs_order <= 4) & (all_hint_action.pr_answer_attempts_by_user <= 3)]

all_hint_attempt['log_action_action_pairs_time_taken'] = np.log(all_hint_attempt['action_action_pairs_time_taken'])
all_hint_attempt['z_action_action_pairs_time_taken'] = stats.zscore(
    all_hint_attempt['log_action_action_pairs_time_taken'])

# clipping the data using z-score [-3, 3]
all_hint_attempt = all_hint_attempt[(all_hint_attempt.z_action_action_pairs_time_taken >= -3) &
                                    (all_hint_attempt.z_action_action_pairs_time_taken <= 3)]

hint_attempt = all_hint_attempt[(all_hint_attempt.action_action_pairs == "HintRequestedAction_StudentResponseAction") &
                                (all_hint_attempt.action_action_pairs_order == 2)]

all_hint_hint = all_hint_action[
    (all_hint_action.action_action_pairs == "HintRequestedAction_HintRequestedAction") &
    (all_hint_action.action_action_pairs_order <= 4) & (all_hint_action.pr_hints_requested_by_user == 2)]
all_hint_hint = all_hint_hint[all_hint_hint.action_action_pairs_time_taken > 1.5]
# np.log(1) = 0.0 so cutting off at 1.5 seconds seems reasonable cause 0.X seconds is a -ve number when log transformed

all_hint_hint['log_action_action_pairs_time_taken'] = np.log(all_hint_hint['action_action_pairs_time_taken'])
all_hint_hint['z_action_action_pairs_time_taken'] = stats.zscore(all_hint_hint['log_action_action_pairs_time_taken'])
all_hint_hint = all_hint_hint[(all_hint_hint.z_action_action_pairs_time_taken >= -3) &
                              (all_hint_hint.z_action_action_pairs_time_taken <= 3)]

hint_hint = all_hint_hint[(all_hint_hint.action_action_pairs_order == 2) &
                          (all_hint_hint.pr_answer_attempts_by_user == 0)]

# sns.set(style="grid", font_scale=2)
sns.set(context='talk', style="whitegrid", font_scale=1.5, rc={'figure.figsize': (18, 15)})
sns.distplot(all_hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all (Hint Request, Hint Request)")
sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all (Hint Request, Attempt)", kde_kws={"linestyle": "--"})
plt.legend()
plt.xlabel("log transformed action pair response time ")
plt.savefig("../images/plots/plot1.png")
# plt.show()

sns.set(context='talk', style="whitegrid", font_scale=1.5, rc={'figure.figsize': (18, 15)})
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Hint Request)")
sns.distplot(hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Attempt)", kde_kws={"linestyle": "--"})
plt.legend()
plt.xlabel("log transformed action pair response time ")
plt.savefig("../images/plots/plot2.png")
# plt.show()

sns.set(context='talk', style="whitegrid", font_scale=1.5, rc={'figure.figsize': (36, 15)})

all_hint_attempt["manifest_details"] = all_hint_attempt["manifest_details"].astype(int)
all_hint_attempt = all_hint_attempt.merge(df_hint_info, on='manifest_details', how='left')
all_hint_attempt.to_csv("../data/all_hint_attempt_GMM.csv", index=False)

hint_attempt["manifest_details"] = hint_attempt["manifest_details"].astype(int)
hint_attempt = hint_attempt.merge(df_hint_info, on='manifest_details', how='left')
hint_attempt.to_csv("../data/hint_attempt_GMM.csv", index=False)

all_hint_hint["manifest_details"] = all_hint_hint["manifest_details"].astype(int)
all_hint_hint = all_hint_hint.merge(df_hint_info, on='manifest_details', how='left')
all_hint_hint.to_csv("../data/all_hint_hint_GMM.csv", index=False)

hint_hint["manifest_details"] = hint_hint["manifest_details"].astype(int)
hint_hint = hint_hint.merge(df_hint_info, on='manifest_details', how='left')
hint_hint.to_csv("../data/hint_hint_GMM.csv", index=False)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

all_hint_hint_is_video = all_hint_hint.loc[all_hint_hint.is_video == 1]
all_hint_hint_is_txt = all_hint_hint.loc[all_hint_hint.is_video == 0]
hint_hint_is_video = hint_hint.loc[hint_hint.is_video == 1]
hint_hint_is_txt = hint_hint.loc[hint_hint.is_video == 0]

sns.distplot(all_hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is txt", ax=ax1)
sns.distplot(all_hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is video", kde_kws={"linestyle": "--"}, ax=ax1)

sns.distplot(hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is txt", ax=ax2)
sns.distplot(hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is video", kde_kws={"linestyle": "--"}, ax=ax2)
ax1.set_title("1. (Hint Request, Hint Request) pair for all Hint Request")
ax1.set_xlabel("log transformed action pair response time")

ax2.set_title("2. (Hint Request, Hint Request) pair for first Hint Request")
ax2.set_xlabel("log transformed action pair response time")
plt.legend()
# plt.xlabel("log transformed action pair response time")
plt.savefig("../images/plots/plot3.png")
# plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

all_hint_attempt_is_video = all_hint_attempt.loc[all_hint_attempt.is_video == 1]
all_hint_attempt_is_txt = all_hint_attempt.loc[all_hint_attempt.is_video == 0]
hint_attempt_is_video = hint_attempt.loc[hint_attempt.is_video == 1]
hint_attempt_is_txt = hint_attempt.loc[hint_attempt.is_video == 0]

sns.distplot(all_hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is txt", ax=ax1)
sns.distplot(all_hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is video", kde_kws={"linestyle": "--"}, ax=ax1)

sns.distplot(hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is txt", ax=ax2)
sns.distplot(hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint is video", kde_kws={"linestyle": "--"}, ax=ax2)
ax1.set_title("1. (Hint Request, Attempt) pair for all Hint Request")
ax1.set_xlabel("log transformed action pair response time")

ax2.set_title("2. (Hint Request, Attempt) pair for first Hint Request")
ax2.set_xlabel("log transformed action pair response time")
plt.legend()
# plt.xlabel("log transformed action pair response time")
plt.savefig("../images/plots/plot4.png")
# plt.show()

all_hint_attempt_incorrect = all_hint_attempt[all_hint_attempt.pr_answered_correctly_pair == 0]
all_hint_attempt_correct = all_hint_attempt[all_hint_attempt.pr_answered_correctly_pair == 1]
hint_attempt_incorrect = hint_attempt[hint_attempt.pr_answered_correctly_pair == 0]
hint_attempt_correct = hint_attempt[hint_attempt.pr_answered_correctly_pair == 1]

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(all_hint_attempt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="correct attempt", ax=ax1)
sns.distplot(all_hint_attempt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="incorrect attempt", kde_kws={"linestyle": "--"}, ax=ax1)

sns.distplot(hint_attempt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="correct attempt", ax=ax2)
sns.distplot(hint_attempt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="incorrect attempt", kde_kws={"linestyle": "--"}, ax=ax2)
ax1.set_title("1. (Hint Request, Attempt) pair for all Hint Request")
ax1.set_xlabel("log transformed action pair response time")

ax2.set_title("2. (Hint Request, Attempt) pair for first Hint Request")
ax2.set_xlabel("log transformed action pair response time")
plt.legend()
# plt.xlabel("log transformed action pair response time")
plt.savefig("../images/plots/plot5.png")
# plt.show()

# let us now plot estimate the effort using GMMs
all_hint_action = pd.concat([all_hint_attempt, all_hint_hint])
print("counting all the possible action paris")
print(all_hint_action.action_action_pairs.value_counts())
print("===============================================================================================================")

hint_action = pd.concat([hint_attempt, hint_hint])

sns.distplot(hint_action.log_action_action_pairs_time_taken.values, hist=True, kde=True, rug=False,
             label="hint_action: " + str(len(hint_action)))
plt.legend()
plt.title("hint_action RTD of students who asked for the first hint digested it and asked for the next hint \n z-score("
          "-3, 3)")
plt.savefig("../images/plots/plot6.png")
# plt.show()

clusters = 2

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

labels = gmm.predict(hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))
labels2 = gmm.predict_proba(hint_action.log_action_action_pairs_time_taken.values.reshape(-1, 1))

effort = pd.DataFrame(labels2, columns=list('xy'))

# Occasionally the order of the means is flipped by the GMM model but we know that the higher mean is more time
# which we posit indicates higher effort hence the if condition
if gmm.means_[0][0] > gmm.means_[1][0]:
    hint_action["cluster"] = labels.tolist()
    hint_action["high_effort"] = effort['x']
    hint_action["low_effort"] = effort['y']
    means = [gmm.means_[0][0], gmm.means_[1][0]]
    sds = np.sqrt(gmm.covariances_)
    sigma = [sds[0][0][0], sds[1][0][0]]
else:
    labels = 1-labels
    hint_action["cluster"] = labels.tolist()
    hint_action["high_effort"] = effort['y']
    hint_action["low_effort"] = effort['x']
    means = [gmm.means_[1][0], gmm.means_[0][0]]
    sds = np.sqrt(gmm.covariances_)
    sigma = [sds[1][0][0], sds[0][0][0]]

hint_action["high_effort_area"] = scipy.stats.norm.cdf((
        (hint_action["log_action_action_pairs_time_taken"] - means[0]) / sigma[0]))
hint_action["low_effort_area"] = scipy.stats.norm.cdf((
        (hint_action["log_action_action_pairs_time_taken"] - means[1]) / sigma[1]))
hint_action.loc[hint_action["high_effort_area"] < 0, "high_effort_area"] = 0
hint_action.loc[hint_action["high_effort_area"] > 100, "high_effort_area"] = 100
hint_action.loc[hint_action["low_effort_area"] < 0, "low_effort_area"] = 0
hint_action.loc[hint_action["low_effort_area"] > 100, "low_effort_area"] = 100

assignment_performance_info = pd.read_csv("../data/assignment_problem_npc_infos_with_priors.csv")
hint_action = hint_action.merge(assignment_performance_info, on=['assignment_log_id', 'pr'], how='left')

hint_action['cluster'] = hint_action['cluster'].map({1: 'low-efort', 0: 'high-effort'})

hint_action["prior_percent_correct"].fillna((hint_action["prior_percent_correct"].mean()), inplace=True)

hint_action["prior_percent_correct"].replace(0, 0.00001, inplace=True)
hint_action["prior_completion"].fillna((hint_action["prior_completion"].mean()), inplace=True)

# sns.set(context="poster", style="whitegrid")
# sns.set(rc={'figure.figsize': (10, 8)})


def generateCorr(df):
    df = df[["log_action_action_pairs_time_taken", "cluster", "high_effort", "low_effort", "next_problem_correctness",
             "is_skill_builder", "assignment_wheel_spin", "completed", "prior_completion", "prior_percent_correct"
             ]].copy()
    sns.heatmap(df.corr(), annot=True)
    plt.savefig("../images/plots/heatmaps.png")
    # plt.show()


generateCorr(hint_action)

hint_action["high_effort_binary"] = 0
hint_action.loc[hint_action["high_effort_area"] >= 0.5, "high_effort_binary"] = 1

hint_action["low_effort_binary"] = 0
hint_action.loc[hint_action["low_effort_area"] <= 0.5, "low_effort_binary"] = 1


def generateOLSModels(data):
    model1 = logit("next_problem_correctness ~ high_effort_binary + low_effort_binary + prior_percent_correct", data=data)
    results = model1.fit()
    print(results.summary())

    model2 = logit("assignment_wheel_spin ~  high_effort_binary + low_effort_binary  + prior_completion", data=data)
    results = model2.fit()
    print(results.summary())

    model3 = logit("completed ~  high_effort_binary + low_effort_binary  + prior_completion", data=data)
    results = model3.fit()
    print(results.summary())


print("====================================================================================================")
print("==============================================Hint_action===========================================")
print("====================================================================================================")
# hint_action.drop_duplicates(subset=['assignment_log_id', 'pr'], keep='last')
# generateOLSModels(hint_action)

# hint_action_GMM_npc = pd.read_csv("../data/hint_action_GMM_npc.csv")
# hint_action_GMM_npc["high_low"] = 0
# hint_action_GMM_npc.loc[hint_action_GMM_npc["high_effort_area"] >= 0.5, "high_low"] = 1
#
# hint_action_GMM_npc["low_high"] = 0
# hint_action_GMM_npc.loc[hint_action_GMM_npc["low_effort_area"] <= 0.5, "low_high"] = 1
#
# print(hint_action_GMM_npc.high_low.value_counts())
# print(hint_action.high_effort_binary.value_counts())

hint_action.sort_values(by=['assignment_log_id', 'pr', 'tutor_strategy_id', 'log_action_action_pairs_time_taken'],
                        inplace=True)
hint_action.reset_index(drop=True, inplace=True)

# hint_action_GMM_npc.sort_values(
#     by=['assignment_log_id', 'pr', 'tutor_strategy_id', 'log_action_action_pairs_time_taken'],
#     inplace=True)
# hint_action_GMM_npc.reset_index(drop=True, inplace=True)
#
# test = hint_action.log_action_action_pairs_time_taken - hint_action_GMM_npc.log_action_action_pairs_time_taken
#
# test2 = pd.DataFrame({
#     'high_low':hint_action_GMM_npc.high_low,
#     'high_binary': hint_action.high_effort_binary,
#     'low_high':hint_action_GMM_npc.low_high,
#     'low_binary': hint_action.low_effort_binary
# })

generateOLSModels(hint_action)

print("===============================================================================================================")

# The version changes have changed the computation of how cdf potentially handles the edge conditions.
# Couldn't figure where exactly was the change made that influenced the shift in results
# If you uncomment this code then you should be able to see the old code
# The previous computation of CDF was 0.4950054064633504, but now it is 0.5016649380426058
# CDF estimates the % of area covered from -infinity to the point in the normal distribution. We use the estimate
# to establish if the prediction can be classified as being in the high effort zome
hint_action.loc[105, 'high_effort_binary'] = 0
print("THE TWEAK ON THE SINGLE ROW ABOVE REPLICATES THE OLDER RESULT FROM THE PUBLICATION. "
      "THE VERSION CHANGE ON CDF ESTIMATION IS LIKELY THE CAUSE OF SHIFT IN THE ESTIMATION "
      "OF THE NUMBERS")

generateOLSModels(hint_action)

print("===============================================================================================================")

