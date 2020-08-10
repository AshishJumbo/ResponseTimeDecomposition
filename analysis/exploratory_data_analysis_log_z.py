import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# NOTE: the same as the other file only instead of randomly winzorising the data we are taking logs and z-scoring
#  them to have an easier time justifying things later on
df_main = pd.read_csv('../data/RTD_data_randomsample_20K_new.csv')

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
sns.set(rc={'figure.figsize': (15, 13)})

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

sns.distplot(df_temp.z_action_action_pairs_time_taken, label="all actions", hist=False, rug=True, color='#FFA500')
sns.distplot(df_temp0.z_action_action_pairs_time_taken, label="all incorrect", hist=False, rug=True, color='r')
sns.distplot(df_temp1.z_action_action_pairs_time_taken, label="all correct", hist=False, rug=True, color='g')
plt.legend()
plt.title("ProblemStartedAction_StudentResponseAction breakdown by correctness")
plt.show()

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

sns.distplot(action_pair_2.z_action_action_pairs_time_taken, label="all actions", hist=False, rug=False,
             color='#EF4444')
sns.distplot(action_pair_2_respond.z_action_action_pairs_time_taken, label="respond again", hist=False, rug=False,
             color='#009f75')
sns.distplot(action_pair_2_answer.z_action_action_pairs_time_taken, label="ask for answer", hist=False, rug=False,
             color='#394BA0')
sns.distplot(action_pair_2_hint.z_action_action_pairs_time_taken, label="ask for hint", hist=False, rug=False,
             color='#0099CC')
sns.distplot(action_pair_2_scaffold.z_action_action_pairs_time_taken, label="ask for scaffold", hist=False, rug=False,
             color='#FAA31B')
sns.distplot(action_pair_2_resumed.z_action_action_pairs_time_taken, label="resumed", hist=False, rug=False,
             color='#000000')
plt.legend()
plt.title("Second action pair breakdown if first action pair was incorrect")
plt.show()

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

sns.distplot(action_hint_hint.z_action_action_pairs_time_taken, label="hint_hint_hint", hist=False, rug=False)
sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken, label="read problem asked for hint 1", hist=False, rug=False)
sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken, label="read hint 1 asked for hint 2", hist=False, rug=False)
sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken, label="read hint 2 asked for hint 3", hist=False, rug=False)
plt.legend()
plt.title("First, second and third hint request z-scored")
plt.show()
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

sns.distplot(action_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
sns.distplot(action_hint_hint_1.action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
sns.distplot(action_hint_hint_2.action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
sns.distplot(action_hint_hint_3.action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
plt.legend()
plt.title("First, second and third hint request actual values")
plt.show()

action_hint_hint = action_hint_hint[(action_hint_hint.z_action_action_pairs_time_taken > -3) &
                                    (action_hint_hint.z_action_action_pairs_time_taken < 3)]

action_hint_hint_1 = action_hint_hint[action_hint_hint.action_action_pairs_order == 1]
action_hint_hint_2 = action_hint_hint[action_hint_hint.action_action_pairs_order == 2]
action_hint_hint_3 = action_hint_hint[action_hint_hint.action_action_pairs_order == 3]

sns.distplot(action_hint_hint.z_action_action_pairs_time_taken, label="hint_hint_hint", hist=False, rug=False)
sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken, label="read problem asked for hint 1", hist=False, rug=False)
sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken, label="read hint 1 asked for hint 2", hist=False, rug=False)
sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken, label="read hint 2 asked for hint 3", hist=False, rug=False)
plt.legend()
plt.title("First, second and third hint request z-scored [-3,3]")
plt.show()

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

sns.distplot(action_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
sns.distplot(action_hint_hint_1.action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
sns.distplot(action_hint_hint_2.action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
sns.distplot(action_hint_hint_3.action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
plt.legend()
plt.title("First, second and third hint request actual values with z-scores [-3, 3]")
plt.show()

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

sns.distplot(action_hint_hint.z_action_action_pairs_time_taken, label="hint_hint_hint", hist=False, rug=False)
sns.distplot(action_hint_hint_1.z_action_action_pairs_time_taken, label="read problem asked for hint 1", hist=False, rug=False)
sns.distplot(action_hint_hint_2.z_action_action_pairs_time_taken, label="read hint 1 asked for hint 2", hist=False, rug=False)
sns.distplot(action_hint_hint_3.z_action_action_pairs_time_taken, label="read hint 2 asked for hint 3", hist=False, rug=False)
plt.legend()
plt.title("First, second and third hint request actual time > 1.5 z-score:[-3,3]")
plt.show()

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

sns.distplot(action_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_hint")
sns.distplot(action_hint_hint_1.action_action_pairs_time_taken.values, kde=True, rug=False, label="read problem asked for hint 1")
sns.distplot(action_hint_hint_2.action_action_pairs_time_taken.values, kde=True, rug=False, label="read hint 1 asked for hint 2")
sns.distplot(action_hint_hint_3.action_action_pairs_time_taken.values, label="read hint 2 asked for hint 3", kde=True, rug=False)
plt.legend()
plt.title("First, second and third hint request actual time > 1.5 z-score:[-3,3], actual values")
plt.show()

print("================================================================================================================")
print("=breaking down the first hint_hint pair as it gives us an understanding of the student digesting the first hint=")
print("================================================================================================================")

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

sns.distplot(all_hint_hint_.z_action_action_pairs_time_taken, label="hint_hint_beakdown_upto2_attempts", hist=False, rug=False)
sns.distplot(hint_hint.z_action_action_pairs_time_taken, label="hint_hint", hist=False, rug=False)
sns.distplot(attempt_hint_hint.z_action_action_pairs_time_taken, label="attempt_hint_hint", hist=False, rug=False)
sns.distplot(attempt_attempt_hint_hint.z_action_action_pairs_time_taken, label="attempt_attempt_hint_hint", hist=False, rug=False)
plt.legend()
plt.title("Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5")
plt.show()

all_hint_hint_ = all_hint_hint_[(all_hint_hint_.z_action_action_pairs_time_taken > -3) &
                                    (all_hint_hint_.z_action_action_pairs_time_taken < 3)]

sns.distplot(all_hint_hint_.z_action_action_pairs_time_taken, label="hint_hint_beakdown_upto2_attempts", hist=False, rug=False)
sns.distplot(hint_hint.z_action_action_pairs_time_taken, label="hint_hint", hist=False, rug=False)
sns.distplot(attempt_hint_hint.z_action_action_pairs_time_taken, label="attempt_hint_hint", hist=False, rug=False)
sns.distplot(attempt_attempt_hint_hint.z_action_action_pairs_time_taken, label="attempt_attempt_hint_hint", hist=False, rug=False)
plt.legend()
plt.title("Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5, "
          "z-score[-3,3]")
plt.show()

sns.distplot(all_hint_hint_.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint_beakdown_upto2_attempts")
sns.distplot(hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="hint_hint")
sns.distplot(attempt_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="attempt_hint_hint")
sns.distplot(attempt_attempt_hint_hint.action_action_pairs_time_taken.values, kde=True, rug=False, label="attempt_attempt_hint_hint")
plt.legend()
plt.title("Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in actual time > 1.5, "
          "z-score[-3,3]")
plt.show()




