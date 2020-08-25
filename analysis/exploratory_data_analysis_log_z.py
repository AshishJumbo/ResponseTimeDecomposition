import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

# NOTE: the same as the other file only instead of randomly winzorising the data we are taking logs and z-scoring
#  them to have an easier time justifying things later on
df_main = pd.read_csv('../data/RTD_data_randomsample_20K_new.csv')
df_hint_info = pd.read_csv("../data/hint_infos.csv")

print(df_main.describe())

# # PS level action don't have a PR associated with them
# # df_main = df_main[df_main.pr != -1]
#
# # the ProblemFinishedAction info is accounted for by the StudentResponseAction_ProblemFinishedAction pair so dropping
# # them as well The StudentResponseAction_ProblemFinishedAction action itself is a automated action sequence pair
# # probably should be dropping that as well
#
# df_main = df_main[df_main.action_action_pairs != "ProblemFinishedAction"]

df_main.action_action_pairs_time_taken.replace({0: 0.000000001}, inplace=True)
# df_main['log_action_action_pairs_time_taken'] = np.log(df_main['action_action_pairs_time_taken'])
# df_main['z_action_action_pairs_time_taken'] = stats.zscore(df_main['log_action_action_pairs_time_taken'])

sns.set(style="whitegrid")
sns.set(rc={'figure.figsize': (16, 12)})

# sns.distplot(df_main.z_action_action_pairs_time_taken.values, kde=True, rug=False)
# plt.title("histogram: z-score plt")
# plt.show()

# automated computer generated actions that don't convey much information are getting dropped
df_main = df_main[df_main.action_action_pairs != "ProblemFinishedAction_ProblemSetFinishedAction"]
# problemset level action so it's significance is already reflected by the
# StudentResponseAction_UserSelectedContinueAction
df_main = df_main[df_main.action_action_pairs != "UserSelectedContinueAction"]
# dropping all ps level actions because the pr level pairing already captures these.
df_main = df_main[df_main.pr != -1]


def plot_distribution(indices):
    df_temp = df_main[df_main.action_action_pairs.isin(indices)]
    df_temp['log_action_action_pairs_time_taken'] = np.log(df_temp['action_action_pairs_time_taken'])
    df_temp['z_action_action_pairs_time_taken'] = stats.zscore(df_temp['log_action_action_pairs_time_taken'])
    sns.distplot(df_temp.z_action_action_pairs_time_taken.values, kde=True, rug=False)
    plt.legend()
    # median = round(df_temp.action_action_pairs_time_taken.median(), 2)
    # mean = round(df_temp.action_action_pairs_time_taken.mean(), 2)
    # mode = round(df_temp.action_action_pairs_time_taken.mode(), 2)
    # plt.title("z-score distrbn plt for: " + ' & '.join(indices) + " with mean : " + str(mean) + " with median : " + str(median) + " with mode : " + str(mode))
    # plt.show()


# indices = df_main.action_action_pairs.value_counts().index.tolist()
# filter_indices = [indices for ]
#
# for index in indices:
#     plot_distribution(index)


# NOTE: The most common pair is ProblemStartedAction_StudentResponseAction
#  let's decompose these first based on whether they are correct or not
#  and then based on if they seek hint if they are incorrect

df_temp = df_main[df_main.action_action_pairs == "ProblemStartedAction_StudentResponseAction"]
df_temp['log_action_action_pairs_time_taken'] = np.log(df_temp['action_action_pairs_time_taken'])
df_temp['z_action_action_pairs_time_taken'] = stats.zscore(df_temp['log_action_action_pairs_time_taken'])

df_temp0 = df_temp[df_temp.pr_answered_correctly_pair == 0]
df_temp1 = df_temp[df_temp.pr_answered_correctly_pair == 1]

# sns.distplot(df_temp.z_action_action_pairs_time_taken, label="all actions", hist=False, rug=True, color='#FFA500')
# sns.distplot(df_temp0.z_action_action_pairs_time_taken, label="all incorrect", hist=False, rug=True, color='r')
# sns.distplot(df_temp1.z_action_action_pairs_time_taken, label="all correct", hist=False, rug=True, color='g')
# plt.legend()
# plt.title(" 1. ProblemStartedAction_StudentResponseAction breakdown by correctness")
# plt.show()

# NOTE: tryng to decompose the response time after the first response was incorrect

action_pair_2 = df_main[(df_main.action_action_pairs_order == 2) & (df_main.action_type == "StudentResponseAction") &
                        (~df_main.action_action_pairs.isin(["StudentResponseAction_UserSelectedContinueAction",
                                                            "StudentResponseAction_ProblemSetFinishedAction",
                                                            "StudentResponseAction_ProblemSetMasteredAction",
                                                            "StudentResponseAction_ProblemSetStartedAction",
                                                            "StudentResponseAction_ProblemLimitExceededAction"]))]

action_pair_2['log_action_action_pairs_time_taken'] = np.log(action_pair_2['action_action_pairs_time_taken'])
action_pair_2['z_action_action_pairs_time_taken'] = stats.zscore(action_pair_2['log_action_action_pairs_time_taken'])

action_pair_2_respond = action_pair_2[
    action_pair_2.action_action_pairs == "StudentResponseAction_StudentResponseAction"]
action_pair_2_answer = action_pair_2[action_pair_2.action_action_pairs == "StudentResponseAction_AnswerRequestedAction"]
action_pair_2_hint = action_pair_2[action_pair_2.action_action_pairs == "StudentResponseAction_HintRequestedAction"]
action_pair_2_resumed = action_pair_2[
    action_pair_2.action_action_pairs == "StudentResponseAction_ProblemSetResumedAction"]
action_pair_2_scaffold = action_pair_2[
    action_pair_2.action_action_pairs == "StudentResponseAction_ScaffoldingRequestedAction"]

# sns.distplot(action_pair_2.z_action_action_pairs_time_taken, label="all actions", hist=False, rug=False,
#              color='#EF4444')
# sns.distplot(action_pair_2_respond.z_action_action_pairs_time_taken, label="respond again", hist=False, rug=False,
#              color='#009f75')
# sns.distplot(action_pair_2_answer.z_action_action_pairs_time_taken, label="ask for answer", hist=False, rug=False,
#              color='#394BA0')
# sns.distplot(action_pair_2_hint.z_action_action_pairs_time_taken, label="ask for hint", hist=False, rug=False,
#              color='#0099CC')
# sns.distplot(action_pair_2_scaffold.z_action_action_pairs_time_taken, label="ask for scaffold", hist=False, rug=False,
#              color='#FAA31B')
# sns.distplot(action_pair_2_resumed.z_action_action_pairs_time_taken, label="resumed", hist=False, rug=False,
#              color='#000000')
# plt.legend()
# plt.title(" 2. Second action pair breakdown if first action pair was incorrect")
# plt.show()

# StudentResponseAction_HintRequestedAction
action_hint_hint = df_main[((df_main.action_action_pairs == "HintRequestedAction_HintRequestedAction") &
                            (df_main.action_action_pairs_order == 3)) |
                           ((df_main.action_action_pairs == "HintRequestedAction_HintRequestedAction") &
                            (df_main.action_action_pairs_order == 2)) |
                           (((df_main.action_action_pairs == "ProblemStartedAction_HintRequestedAction") |
                             (df_main.action_action_pairs == "ProblemResumedAction_HintRequestedAction")) &
                            (df_main.action_action_pairs_order == 1))]

action_hint_hint['log_action_action_pairs_time_taken'] = np.log(action_hint_hint['action_action_pairs_time_taken'])
action_hint_hint['z_action_action_pairs_time_taken'] = stats.zscore(
    action_hint_hint['log_action_action_pairs_time_taken'])

action_hint_hint_1 = action_hint_hint[action_hint_hint.action_action_pairs_order == 1]
action_hint_hint_2 = action_hint_hint[action_hint_hint.action_action_pairs_order == 2]
action_hint_hint_3 = action_hint_hint[action_hint_hint.action_action_pairs_order == 3]

# sns.distplot(action_hint_hint.z_action_action_pairs_time_taken, label="hint_hint_hint", hist=False, rug=False)
# sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken, label="read problem asked for hint 1", hist=False,
#              rug=False)
# sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken, label="read hint 1 asked for hint 2", hist=False,
#              rug=False)
# sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken, label="read hint 2 asked for hint 3", hist=False,
#              rug=False)
# plt.legend()
# plt.title(" 3. First, second and third hint request z-scored")
# plt.show()
#
# sns.distplot(action_hint_hint.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
# plt.legend()
# plt.show()

# sns.distplot(action_hint_hint.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.log_action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
# plt.legend()
# plt.show()

# sns.distplot(action_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True,
#              rug=False)
# plt.legend()
# plt.title(" 4. First, second and third hint request actual values")
# plt.show()

action_hint_hint = action_hint_hint[(action_hint_hint.z_action_action_pairs_time_taken > -3) &
                                    (action_hint_hint.z_action_action_pairs_time_taken < 3)]

action_hint_hint_1 = action_hint_hint[action_hint_hint.action_action_pairs_order == 1]
action_hint_hint_2 = action_hint_hint[action_hint_hint.action_action_pairs_order == 2]
action_hint_hint_3 = action_hint_hint[action_hint_hint.action_action_pairs_order == 3]

# sns.distplot(action_hint_hint.z_action_action_pairs_time_taken, label="hint_hint_hint", hist=False, rug=False)
# sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken, label="read problem asked for hint 1", hist=False,
#              rug=False)
# sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken, label="read hint 1 asked for hint 2", hist=False,
#              rug=False)
# sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken, label="read hint 2 asked for hint 3", hist=False,
#              rug=False)
# plt.legend()
# plt.title(" 5. First, second and third hint request z-scored [-3,3]")
# plt.show()

# sns.distplot(action_hint_hint.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
# plt.legend()
# plt.show()
#
# sns.distplot(action_hint_hint.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.log_action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
# plt.legend()
# plt.show()

# sns.distplot(action_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True,
#              rug=False)
# plt.legend()
# plt.title(" 6. First, second and third hint request actual values with z-scores [-3, 3]")
# plt.show()

# NOTE: forgot to include the situations where the hintRequest was the second action but first request
# NOTE: should I be removing the records that are less than 1.5 seconds
# NOTE Question: Should we do it per action level or per hint order level right now it is done per action level
#  because the assumption is to do a response time decomposition of actions

action_hint_hint = df_main[((df_main.action_action_pairs == "HintRequestedAction_HintRequestedAction") &
                            (df_main.action_action_pairs_order == 3)) |
                           ((df_main.action_action_pairs == "HintRequestedAction_HintRequestedAction") &
                            (df_main.action_action_pairs_order == 2)) |
                           (((df_main.action_action_pairs == "ProblemStartedAction_HintRequestedAction") |
                             (df_main.action_action_pairs == "ProblemResumedAction_HintRequestedAction")) &
                            (df_main.action_action_pairs_order == 1))]

action_hint_hint = action_hint_hint[action_hint_hint.action_action_pairs_time_taken > 1.5]

action_hint_hint['log_action_action_pairs_time_taken'] = np.log(action_hint_hint['action_action_pairs_time_taken'])
action_hint_hint['z_action_action_pairs_time_taken'] = stats.zscore(
    action_hint_hint['log_action_action_pairs_time_taken'])

action_hint_hint = action_hint_hint[(action_hint_hint.z_action_action_pairs_time_taken > -3) &
                                    (action_hint_hint.z_action_action_pairs_time_taken < 3)]

action_hint_hint_1 = action_hint_hint[action_hint_hint.action_action_pairs_order == 1]
action_hint_hint_2 = action_hint_hint[action_hint_hint.action_action_pairs_order == 2]
action_hint_hint_3 = action_hint_hint[action_hint_hint.action_action_pairs_order == 3]

# sns.distplot(action_hint_hint.z_action_action_pairs_time_taken, label="hint_hint_hint", hist=False, rug=False)
# sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken, label="read problem asked for hint 1", hist=False,
#              rug=False)
# sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken, label="read hint 1 asked for hint 2", hist=False,
#              rug=False)
# sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken, label="read hint 2 asked for hint 3", hist=False,
#              rug=False)
# plt.legend()
# plt.title(" 7. First, second and third hint request actual time > 1.5 z-score:[-3,3]")
# plt.show()

# sns.distplot(action_hint_hint.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
# plt.legend()
# plt.show()
#
# sns.distplot(action_hint_hint.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.log_action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.log_action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
# plt.legend()
# plt.show()

# sns.distplot(action_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
# sns.distplot(action_hint_hint_1.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="read problem asked for hint 1")
# sns.distplot(action_hint_hint_2.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="read hint 1 asked for hint 2")
# sns.distplot(action_hint_hint_3.action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True,
#              rug=False)
# plt.legend()
# plt.title(" 8. First, second and third hint request actual time > 1.5 z-score:[-3,3], actual values")
# plt.show()

print(
    "================================================================================================================")
print(
    "=breaking down the first hint_hint pair as it gives us an understanding of the student digesting the first hint=")
print(
    "================================================================================================================")

all_hint_hint_ = df_main[(df_main.action_action_pairs == "HintRequestedAction_HintRequestedAction") &
                         (df_main.action_action_pairs_order <= 4) & (df_main.pr_hints_requested_by_user == 2)]
all_hint_hint_ = all_hint_hint_[all_hint_hint_.action_action_pairs_time_taken > 1.5]

all_hint_hint_['log_action_action_pairs_time_taken'] = np.log(all_hint_hint_['action_action_pairs_time_taken'])
all_hint_hint_['z_action_action_pairs_time_taken'] = stats.zscore(
    all_hint_hint_['log_action_action_pairs_time_taken'])

hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 2) &
                           (all_hint_hint_.pr_answer_attempts_by_user == 0)]
attempt_hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 3) &
                                   (all_hint_hint_.pr_answer_attempts_by_user == 1)]
attempt_attempt_hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 4) &
                                           (all_hint_hint_.pr_answer_attempts_by_user == 2)]

# sns.distplot(all_hint_hint_.z_action_action_pairs_time_taken, label="hint_hint_beakdown_upto2_attempts", hist=False,
#              rug=False)
# sns.distplot(hint_hint.z_action_action_pairs_time_taken, label="hint_hint", hist=False, rug=False)
# sns.distplot(attempt_hint_hint.z_action_action_pairs_time_taken, label="attempt_hint_hint", hist=False, rug=False)
# sns.distplot(attempt_attempt_hint_hint.z_action_action_pairs_time_taken, label="attempt_attempt_hint_hint", hist=False,
#              rug=False)
# plt.legend()
# plt.title(" 9. Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5")
# plt.show()

all_hint_hint_ = all_hint_hint_[(all_hint_hint_.z_action_action_pairs_time_taken > -3) &
                                (all_hint_hint_.z_action_action_pairs_time_taken < 3)]

hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 2) &
                           (all_hint_hint_.pr_answer_attempts_by_user == 0)]
attempt_hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 3) &
                                   (all_hint_hint_.pr_answer_attempts_by_user == 1)]
attempt_attempt_hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 4) &
                                           (all_hint_hint_.pr_answer_attempts_by_user == 2)]

# sns.distplot(all_hint_hint_.z_action_action_pairs_time_taken, label="hint_hint_beakdown_upto2_attempts", hist=False,
#              rug=False)
# sns.distplot(hint_hint.z_action_action_pairs_time_taken, label="hint_hint count: " + str(len(hint_hint)), hist=False,
#              rug=False)
# sns.distplot(attempt_hint_hint.z_action_action_pairs_time_taken,
#              label="attempt_hint_hint: " + str(len(attempt_hint_hint)), hist=False, rug=False)
# sns.distplot(attempt_attempt_hint_hint.z_action_action_pairs_time_taken,
#              label="attempt_attempt_hint_hint: " + str(len(attempt_attempt_hint_hint)),
#              hist=False, rug=False)
# plt.legend()
# plt.title("10. Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5, "
#           "z-score[-3,3]")
# plt.show()

# sns.distplot(all_hint_hint_.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="hint_hint_beakdown_upto2_attempts")
# sns.distplot(hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="hint_hint: " + str(len(hint_hint)))
# sns.distplot(attempt_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="attempt_hint_hint: " + str(len(attempt_hint_hint)))
# sns.distplot(attempt_attempt_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="attempt_attempt_hint_hint: " + str(len(attempt_attempt_hint_hint)))
# plt.legend()
# plt.title("11. Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5, "
#           "z-score[-3,3]")
# plt.show()

print("==================================================================================================")
print("=====================Exploring the SKlearn Gaussian Mixture modelling library=====================")
print("==================================================================================================")

from sklearn.cluster import KMeans

kmeans = KMeans().fit(all_hint_hint_.z_action_action_pairs_time_taken.values.reshape(-1, 1))

clusters = 3

gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    all_hint_hint_.z_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

# NOTE: next steps?
#  attempt_hint_attempt
#  hint_attempt
#  these will contrast against the hint_hint and attempt_hint_hint to give us an insight into how long
#  students take to formulate a response
# RESEARCH QUESTION: depending on how we formulate our definition of effort it might help us differentiate
#  the behavioural mannerisms of students exhibiting effort vs students not exhibiting effort while
#  attempting the problem

all_hint_attempt = df_main[df_main.action_action_pairs.str.contains("HintRequestedAction_")]
print(all_hint_attempt.action_action_pairs.value_counts())
all_hint_attempt = all_hint_attempt[
    (all_hint_attempt.action_action_pairs == "HintRequestedAction_StudentResponseAction") &
    (all_hint_attempt.action_action_pairs_order <= 4) &
    (all_hint_attempt.pr_answer_attempts_by_user <= 3)]

all_hint_attempt['log_action_action_pairs_time_taken'] = np.log(all_hint_attempt['action_action_pairs_time_taken'])
all_hint_attempt['z_action_action_pairs_time_taken'] = stats.zscore(all_hint_attempt[
                                                                        'log_action_action_pairs_time_taken'])

hint_attempt = all_hint_attempt[(all_hint_attempt.action_action_pairs == "HintRequestedAction_StudentResponseAction") &
                                (all_hint_attempt.action_action_pairs_order == 2)]

attempt_hint_attempt = all_hint_attempt[(all_hint_attempt.action_action_pairs ==
                                         "HintRequestedAction_StudentResponseAction") &
                                        (all_hint_attempt.action_action_pairs_order == 3) &
                                        (all_hint_attempt.pr_answer_attempts_by_user == 2)]

attempt_attempt_hint_attempt = all_hint_attempt[(all_hint_attempt.action_action_pairs ==
                                                 "HintRequestedAction_StudentResponseAction") &
                                                (all_hint_attempt.action_action_pairs_order == 4) &
                                                (all_hint_attempt.pr_answer_attempts_by_user == 3)]

# sns.distplot(all_hint_attempt.z_action_action_pairs_time_taken,
#              label="hint_attempt breakdown upto 3 total attempts with the last pair being hint_attempt", hist=False,
#              rug=False)
# sns.distplot(hint_attempt.z_action_action_pairs_time_taken, label="hint_attempt", hist=False, rug=False)
# sns.distplot(attempt_hint_attempt.z_action_action_pairs_time_taken, label="attempt_hint_attempt", hist=False, rug=False)
# sns.distplot(attempt_attempt_hint_attempt.z_action_action_pairs_time_taken, label="attempt_attempt_hint_attempt",
#              hist=False,
#              rug=False)
# plt.legend()
# plt.title("12. Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5")
# plt.show()

all_hint_attempt = all_hint_attempt[(all_hint_attempt.z_action_action_pairs_time_taken > -3) &
                                    (all_hint_attempt.z_action_action_pairs_time_taken < 3)]

hint_attempt = all_hint_attempt[(all_hint_attempt.action_action_pairs == "HintRequestedAction_StudentResponseAction") &
                                (all_hint_attempt.action_action_pairs_order == 2)]

attempt_hint_attempt = all_hint_attempt[(all_hint_attempt.action_action_pairs ==
                                         "HintRequestedAction_StudentResponseAction") &
                                        (all_hint_attempt.action_action_pairs_order == 3) &
                                        (all_hint_attempt.pr_answer_attempts_by_user == 2)]

attempt_attempt_hint_attempt = all_hint_attempt[(all_hint_attempt.action_action_pairs ==
                                                 "HintRequestedAction_StudentResponseAction") &
                                                (all_hint_attempt.action_action_pairs_order == 4) &
                                                (all_hint_attempt.pr_answer_attempts_by_user == 3)]

# sns.distplot(all_hint_attempt.z_action_action_pairs_time_taken,
#              label="hint_attempt breakdown upto 3 total attempts with the last pair being hint_attempt", hist=False,
#              rug=False)
# sns.distplot(hint_attempt.z_action_action_pairs_time_taken, label="hint_attempt count: " + str(len(hint_attempt)),
#              hist=False,
#              rug=False)
# sns.distplot(attempt_hint_attempt.z_action_action_pairs_time_taken,
#              label="attempt_hint_attempt: " + str(len(attempt_hint_attempt)), hist=False, rug=False)
# sns.distplot(attempt_attempt_hint_attempt.z_action_action_pairs_time_taken,
#              label="attempt_attempt_hint_attempt: " + str(len(attempt_attempt_hint_attempt)),
#              hist=False, rug=False)
# plt.legend()
# plt.title("13. Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5, "
#           "z-score[-3,3]")
# plt.show()
#
# sns.distplot(all_hint_attempt.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="hint_attempt breakdown upto 3 total attempts with the last pair being hint_attempt")
# sns.distplot(hint_attempt.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="hint_attempt: " + str(len(hint_attempt)))
# sns.distplot(attempt_hint_attempt.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="attempt_hint_attempt: " + str(len(attempt_hint_attempt)))
# sns.distplot(attempt_attempt_hint_attempt.action_action_pairs_time_taken.values, kde=True, rug=False,
#              label="attempt_attempt_hint_attempt: " + str(len(attempt_attempt_hint_attempt)))
# plt.legend()
# plt.title("14. Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5, "
#           "z-score[-3,3]")
# plt.show()

hint_attempt_incorrect = hint_attempt[hint_attempt.pr_answered_correctly_pair == 0]
hint_attempt_correct = hint_attempt[hint_attempt.pr_answered_correctly_pair == 1]

# sns.distplot(hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt: " + str(len(hint_attempt)))
# sns.distplot(hint_attempt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt correct: " + str(len(hint_attempt_correct)), color='g')
# sns.distplot(hint_attempt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt incorrect: " + str(len(hint_attempt_incorrect)), color='r')
# sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint: " + str(len(hint_hint)))
# plt.legend()
# plt.title("15. comparing log of the time taken for the first hint_hint vs hint_attempt vs hint_attempt_correct vs hint_"
#           "attempt_incorrect pairs ")
# plt.show()

attempt_hint_attempt_incorrect = attempt_hint_attempt[attempt_hint_attempt.pr_answered_correctly_pair == 0]
attempt_hint_attempt_correct = attempt_hint_attempt[attempt_hint_attempt.pr_answered_correctly_pair == 1]

# sns.distplot(attempt_hint_attempt.z_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#               label="attempt_hint_attempt: " + str(len(attempt_hint_attempt)))
# sns.distplot(attempt_hint_attempt_correct.z_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#               label="attempt_hint_attempt correct: " + str(len(attempt_hint_attempt_correct)), color='g')
sns.distplot(attempt_hint_attempt_incorrect.z_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="attempt_hint_attempt incorrect: " + str(len(attempt_hint_attempt_incorrect)), color='r')
# sns.distplot(hint_hint.z_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#               label="hint_hint: " + str(len(hint_hint)))
# sns.distplot(attempt_hint_hint.z_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#               label="attempt_hint_hint: " + str(len(attempt_hint_hint)), color='c')
plt.legend()
plt.title("16. comparing z[-3,3] of the time taken for the hint_hint vs attempt_hint_hint vs attempt_hint_attempt vs "
          "\n attempt_hint_attempt_correct vs attempt_hint_attempt_incorrect pairs ")
plt.show()

print("=====================================================================")
print("---------------------------a_h_a_incorrect---------------------------")
print("=====================================================================")
gmm_aha_ic = GaussianMixture(n_components=clusters, max_iter=100).fit(
    attempt_hint_attempt_incorrect.z_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm_aha_ic.means_)
print("=====================================================================")

sns.distplot(attempt_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="attempt_hint_attempt: " + str(len(attempt_hint_attempt)))
sns.distplot(attempt_hint_attempt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="attempt_hint_attempt correct: " + str(len(attempt_hint_attempt_correct)), color='g')
sns.distplot(attempt_hint_attempt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="attempt_hint_attempt incorrect: " + str(len(attempt_hint_attempt_incorrect)), color='r')
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
sns.distplot(attempt_hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="attempt_hint_hint: " + str(len(attempt_hint_hint)), color='c')
plt.legend()
plt.title("17. comparing log of the time taken for the hint_hint vs attempt_hint_hint vs attempt_hint_attempt vs "
          "\n attempt_hint_attempt_correct vs attempt_hint_attempt_incorrect pairs ")
plt.show()

all_hint_attempt_incorrect = all_hint_attempt[all_hint_attempt.pr_answered_correctly_pair == 0]
all_hint_attempt_correct = all_hint_attempt[all_hint_attempt.pr_answered_correctly_pair == 1]

sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt: " + str(len(all_hint_attempt)))
sns.distplot(all_hint_attempt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt incorrect: " + str(len(all_hint_attempt_incorrect)), color='r')
sns.distplot(all_hint_attempt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_correct: " + str(len(all_hint_attempt_correct)), color='g')
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint: " + str(len(all_hint_hint_)))
plt.legend()
plt.title("18. all_hint_hint vs all_hint_attempt vs all_hint_attempt_correct vs all_hint_attempt_incorrect")
plt.show()

print("==========================================merging with hint_info==========================================")
all_hint_hint_["manifest_details"] = all_hint_hint_["manifest_details"].astype(int)
all_hint_hint_ = all_hint_hint_.merge(df_hint_info, on="manifest_details", how="left")

all_hint_attempt["manifest_details"] = all_hint_attempt["manifest_details"].astype(int)
all_hint_attempt = all_hint_attempt.merge(df_hint_info, on="manifest_details", how="left")

hint_hint["manifest_details"] = hint_hint["manifest_details"].astype(int)
hint_hint = hint_hint.merge(df_hint_info, on="manifest_details", how="left")

hint_attempt["manifest_details"] = hint_attempt["manifest_details"].astype(int)
hint_attempt = hint_attempt.merge(df_hint_info, on="manifest_details", how="left")

all_hint_attempt_is_video = all_hint_attempt[all_hint_attempt.is_video == 1]
all_hint_attempt_is_txt = all_hint_attempt[all_hint_attempt.is_video == 0]

sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt: " + str(len(all_hint_attempt)))
sns.distplot(all_hint_attempt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt incorrect: " + str(len(all_hint_attempt_incorrect)), color='r')
sns.distplot(all_hint_attempt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_correct: " + str(len(all_hint_attempt_correct)), color='g')
sns.distplot(all_hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_video: " + str(len(all_hint_attempt_is_video)))
sns.distplot(all_hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_txt: " + str(len(all_hint_attempt_is_txt)))
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint: " + str(len(all_hint_hint_)))
plt.legend()
plt.title("19. all_hint_hint vs all_hint_attempt vs all_hint_attempt_correct vs all_hint_attempt_incorrect vs "
          "all_hint_attempt_video vs all_hint_attempt_txt")
plt.show()

all_hint_attempt_is_video_correct = all_hint_attempt_is_video[all_hint_attempt_is_video.pr_answered_correctly_pair == 1]
all_hint_attempt_is_video_incorrect = all_hint_attempt_is_video[
    all_hint_attempt_is_video.pr_answered_correctly_pair == 0]
all_hint_attempt_is_txt_correct = all_hint_attempt_is_txt[all_hint_attempt_is_txt.pr_answered_correctly_pair == 1]
all_hint_attempt_is_txt_incorrect = all_hint_attempt_is_txt[all_hint_attempt_is_txt.pr_answered_correctly_pair == 0]

hint_attempt_is_txt = hint_attempt[hint_attempt.is_video == 0]
hint_attempt_is_video = hint_attempt[hint_attempt.is_video == 1]

hint_attempt_is_txt_correct = hint_attempt_is_txt[hint_attempt_is_txt.pr_answered_correctly_pair == 1]
hint_attempt_is_txt_incorrect = hint_attempt_is_txt[hint_attempt_is_txt.pr_answered_correctly_pair == 0]
hint_attempt_is_video_correct = hint_attempt_is_video[hint_attempt_is_video.pr_answered_correctly_pair == 1]
hint_attempt_is_video_incorrect = hint_attempt_is_video[hint_attempt_is_video.pr_answered_correctly_pair == 0]

sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt: " + str(len(all_hint_attempt)), color='k')
sns.distplot(all_hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_txt_correct: " + str(len(all_hint_attempt_is_txt_correct)), color='g')
sns.distplot(all_hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_txt_incorrect: " + str(len(all_hint_attempt_is_txt_incorrect)), color='r')
sns.distplot(all_hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_txt: " + str(len(all_hint_attempt_is_txt)), color='b')
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint: " + str(len(all_hint_hint_)))
plt.legend()
plt.title("20. all_hint_hint vs all_hint_attempt_txt vs all_hint_attempt_txt_correct vs all_hint_attempt_txt_incorrect")
plt.show()

sns.distplot(all_hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_txt_correct: " + str(len(all_hint_attempt_is_txt_correct)), color='g')
sns.distplot(all_hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_txt_incorrect: " + str(len(all_hint_attempt_is_txt_incorrect)), color='r')
sns.distplot(all_hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_txt: " + str(len(all_hint_attempt_is_txt)), color='b')
plt.legend()
plt.title(
    "20.1 all_hint_hint vs all_hint_attempt_txt vs all_hint_attempt_txt_correct vs all_hint_attempt_txt_incorrect")
plt.show()

sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt: " + str(len(all_hint_attempt)))
sns.distplot(all_hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_video_correct: " + str(len(all_hint_attempt_is_video_correct)), color='g')
sns.distplot(all_hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_video_incorrect: " + str(len(all_hint_attempt_is_video_incorrect)), color='r')
sns.distplot(all_hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_video: " + str(len(all_hint_attempt_is_video)))
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint: " + str(len(all_hint_hint_)))
plt.legend()
plt.title("21. all_hint_hint vs all_hint_attempt_video vs all_hint_attempt_is_video_correct vs "
          "all_hint_attempt_is_video_incorrect")
plt.show()

sns.distplot(all_hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_video_correct: " + str(len(all_hint_attempt_is_video_correct)), color='g')
sns.distplot(all_hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_video_incorrect: " + str(len(all_hint_attempt_is_video_incorrect)), color='r')
sns.distplot(all_hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_video: " + str(len(all_hint_attempt_is_video)))
plt.legend()
plt.title("21.1 all_hint_hint vs all_hint_attempt_video vs all_hint_attempt_is_video_correct vs "
          "all_hint_attempt_is_video_incorrect")
plt.show()

all_hint_hint_is_video = all_hint_hint_[all_hint_hint_.is_video == 1]
all_hint_hint_is_txt = all_hint_hint_[all_hint_hint_.is_video == 0]
sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint_: " + str(len(all_hint_hint_)))
sns.distplot(all_hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint_is_video: " + str(len(all_hint_hint_is_video)))
sns.distplot(all_hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_hint_is_txt: " + str(len(all_hint_hint_is_txt)))
plt.legend()
plt.title("22. Breaking down all_hint_hint first time by incorporating attempts[depth: 2] as well in \n"
          "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
plt.show()

hint_hint_is_video = hint_hint[hint_hint.is_video == 1]
hint_hint_is_txt = hint_hint[hint_hint.is_video == 0]
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
sns.distplot(hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint_is_video: " + str(len(hint_hint_is_video)))
sns.distplot(hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint_is_txt: " + str(len(hint_hint_is_txt)))
plt.legend()
plt.title("22.1 Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in \n"
          "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
plt.show()

sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint: " + str(len(hint_hint)))
sns.distplot(hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint_is_video: " + str(len(hint_hint_is_video)))
sns.distplot(hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_hint_is_txt: " + str(len(hint_hint_is_txt)))
sns.distplot(hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_attempt_is_txt: " + str(len(hint_attempt_is_txt)))
sns.distplot(hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_attempt_is_video: " + str(len(hint_attempt_is_video)))
plt.legend()
plt.title("22.2 Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in \n"
          "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
plt.show()


def hint_type_breakdown(df_temp, gmm_cluesters, label_str1, label_str2, hist=False):
    gmm = GaussianMixture(n_components=gmm_cluesters, max_iter=100).fit(
        df_temp.z_action_action_pairs_time_taken.values.reshape(-1, 1))
    print(label_str1, gmm.means_)

    sns.distplot(df_temp.z_action_action_pairs_time_taken.values, hist=hist, kde=True,
                 rug=False,
                 label=label_str2 + str(len(df_temp)))
    plt.legend()
    plt.title(label_str1)
    plt.show()


hint_type_breakdown(all_hint_attempt_is_video_correct, 2, "23. Means by sklearn [hint-attempt, video, correct]: ",
                    "23. all_hint_attempt_is_video_correct: ", True)
hint_type_breakdown(all_hint_attempt_is_video, 3, "24. Means by sklearn [hint-attempt, video]: ",
                    "24. all_hint_attempt_is_video: ", True)
hint_type_breakdown(all_hint_attempt_is_video_incorrect, 2, "25. Means by sklearn [hint-attempt, video, incorrect]: ",
                    "25. all_hint_attempt_is_video_incorrect: ", True)

sns.distplot(all_hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_video_correct: " + str(len(all_hint_attempt_is_video_correct)), color='g')
sns.distplot(all_hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_video_incorrect: " + str(len(all_hint_attempt_is_video_incorrect)), color='r')
sns.distplot(all_hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_video: " + str(len(all_hint_attempt_is_video)))
plt.legend()
plt.title("21. all_hint_hint vs all_hint_attempt_video vs all_hint_attempt_is_video_correct vs "
          "all_hint_attempt_is_video_incorrect")
plt.show()

hint_type_breakdown(all_hint_attempt_is_txt_correct, 1, "26. Means by sklearn [hint-attempt, text, correct]: ",
                    "26. all_hint_attempt_is_text_correct: ", True)
hint_type_breakdown(all_hint_attempt_is_txt, 3, "27. Means by sklearn [hint-attempt, text]: ",
                    "27. all_hint_attempt_is_text: ", True)
hint_type_breakdown(all_hint_attempt_is_txt_incorrect, 2, "28. Means by sklearn [hint-attempt, text, incorrect]: ",
                    "28. all_hint_attempt_is_text_incorrect: ", True)

sns.distplot(all_hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False, label="all_hint_attempt_is_txt_correct: " + str(len(all_hint_attempt_is_txt_correct)),
             color='g')
sns.distplot(all_hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="all_hint_attempt_is_txt_incorrect: " + str(len(all_hint_attempt_is_txt_incorrect)), color='r')
sns.distplot(all_hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="all_hint_attempt_is_txt: " + str(len(all_hint_attempt_is_txt)), color='b')
plt.legend()
plt.title("20. all_hint_hint vs all_hint_attempt_txt vs all_hint_attempt_txt_correct vs all_hint_attempt_txt_incorrect")
plt.show()

sns.distplot(hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False, label="hint_attempt_is_txt_correct: " + str(len(hint_attempt_is_txt_correct)),
             color='g')
sns.distplot(hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="hint_attempt_is_txt_incorrect: " + str(len(hint_attempt_is_txt_incorrect)), color='r')
sns.distplot(hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_attempt_is_txt: " + str(len(hint_attempt_is_txt)), color='b')
plt.legend()
plt.title("29. hint_attempt_txt vs hint_attempt_txt_correct vs hint_attempt_txt_incorrect")
plt.show()

sns.distplot(hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False, label="hint_attempt_is_video_correct: " + str(len(hint_attempt_is_video_correct)),
             color='g')
sns.distplot(hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
             rug=False,
             label="hint_attempt_is_video_incorrect: " + str(len(hint_attempt_is_video_incorrect)), color='r')
sns.distplot(hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="hint_attempt_is_video: " + str(len(hint_attempt_is_video)), color='b')
plt.legend()
plt.title("30. hint_attempt_video vs hint_attempt_video_correct vs hint_attempt_video_incorrect")
plt.show()

sns.distplot(hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=True, kde=True,
             rug=False, label="hint_attempt_is_video_correct: " + str(len(hint_attempt_is_video_correct)),
             color='g')
sns.distplot(hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=True, kde=True,
             rug=False,
             label="hint_attempt_is_video_incorrect: " + str(len(hint_attempt_is_video_incorrect)), color='r')
plt.legend()
plt.title("31.1 hint_attempt_video vs hint_attempt_video_correct vs hint_attempt_video_incorrect")
plt.show()

print("============================================================================================")
