import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from bs4 import BeautifulSoup
import re

# NOTE: the same as the other file only instead of randomly winzorising the data we are taking logs and z-scoring
#  them to have an easier time justifying things later on
df_main = pd.read_csv('../data/RTD_data_randomsample_20K_new.csv')
df_hint_info = pd.read_csv("../data/hint_infos.csv")


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
    df_hint_info["hint_body_word_count2"] = df_hint_info.hint_body_clean2.str.split().str.len() + \
                                            df_hint_info.hint_body_td_count + df_hint_info.hint_body_wiris_count


hint_body_clean_script()
print(df_main.describe())

# PS level action don't have a PR associated with them
df_main.action_action_pairs_time_taken.replace({0: 0.000000001}, inplace=True)

sns.set(style="whitegrid")
sns.set(rc={'figure.figsize': (18, 15)})

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


# NOTE: The most common pair is ProblemStartedAction_StudentResponseAction
#  let's decompose these first based on whether they are correct or not
#  and then based on if they seek hint if they are incorrect

df_temp = df_main[df_main.action_action_pairs == "ProblemStartedAction_StudentResponseAction"]
df_temp['log_action_action_pairs_time_taken'] = np.log(df_temp['action_action_pairs_time_taken'])
df_temp['z_action_action_pairs_time_taken'] = stats.zscore(df_temp['log_action_action_pairs_time_taken'])

df_temp0 = df_temp[df_temp.pr_answered_correctly_pair == 0]
df_temp1 = df_temp[df_temp.pr_answered_correctly_pair == 1]

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

# NOTE: StudentResponseAction_HintRequestedAction
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

action_hint_hint = action_hint_hint[(action_hint_hint.z_action_action_pairs_time_taken > -3) &
                                    (action_hint_hint.z_action_action_pairs_time_taken < 3)]

action_hint_hint_1 = action_hint_hint[action_hint_hint.action_action_pairs_order == 1]
action_hint_hint_2 = action_hint_hint[action_hint_hint.action_action_pairs_order == 2]
action_hint_hint_3 = action_hint_hint[action_hint_hint.action_action_pairs_order == 3]

# NOTE: forgot to include the situations where the hintRequest was the second action but first request
# NOTE: should I be removing the records that are less than 1.5 seconds

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

all_hint_hint_ = all_hint_hint_[(all_hint_hint_.z_action_action_pairs_time_taken > -3) &
                                (all_hint_hint_.z_action_action_pairs_time_taken < 3)]

hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 2) &
                           (all_hint_hint_.pr_answer_attempts_by_user == 0)]
attempt_hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 3) &
                                   (all_hint_hint_.pr_answer_attempts_by_user == 1)]
attempt_attempt_hint_hint = all_hint_hint_[(all_hint_hint_.action_action_pairs_order == 4) &
                                           (all_hint_hint_.pr_answer_attempts_by_user == 2)]

print("==================================================================================================")
print("=====================Exploring the SKlearn Gaussian Mixture modelling library=====================")
print("==================================================================================================")
clusters = 3
gmm = GaussianMixture(n_components=clusters, max_iter=100).fit(
    all_hint_hint_.z_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm.means_)

all_hint_attempt = df_main[df_main.action_action_pairs.str.contains("HintRequestedAction_")]
# print(all_hint_attempt.action_action_pairs.value_counts())
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

hint_attempt_incorrect = hint_attempt[hint_attempt.pr_answered_correctly_pair == 0]
hint_attempt_correct = hint_attempt[hint_attempt.pr_answered_correctly_pair == 1]

sns.distplot(hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Attempt)", kde_kws={"linestyle": "--"})
sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Hint Request)")
plt.legend()
plt.title("log transformed action pair response time ")
plt.show()

attempt_hint_attempt_incorrect = attempt_hint_attempt[attempt_hint_attempt.pr_answered_correctly_pair == 0]
attempt_hint_attempt_correct = attempt_hint_attempt[attempt_hint_attempt.pr_answered_correctly_pair == 1]

print("=====================================================================")
print("---------------------------a_h_a_incorrect---------------------------")
print("=====================================================================")
gmm_aha_ic = GaussianMixture(n_components=clusters, max_iter=100).fit(
    attempt_hint_attempt_incorrect.z_action_action_pairs_time_taken.values.reshape(-1, 1))
print("Means by sklearn: ", gmm_aha_ic.means_)
print("=====================================================================")

all_hint_attempt_incorrect = all_hint_attempt[all_hint_attempt.pr_answered_correctly_pair == 0]
all_hint_attempt_correct = all_hint_attempt[all_hint_attempt.pr_answered_correctly_pair == 1]

# sns.set(font_scale=1.2)
sns.set(context="talk", style="whitegrid", font_scale=2)

# sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt: " + str(len(all_hint_attempt)))
# sns.distplot(all_hint_attempt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt incorrect: " + str(len(all_hint_attempt_incorrect)), color='r')
# sns.distplot(all_hint_attempt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_correct: " + str(len(all_hint_attempt_correct)), color='g')
# sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint: " + str(len(hint_hint)))
# sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint: " + str(len(all_hint_hint_)))
# plt.legend()
# plt.title("18. all_hint_hint vs all_hint_attempt vs all_hint_attempt_correct vs all_hint_attempt_incorrect")
# plt.show()

print("==========================================merging with hint_info==========================================")
all_hint_hint_["manifest_details"] = all_hint_hint_["manifest_details"].astype(int)
all_hint_hint_ = all_hint_hint_.merge(df_hint_info, on="manifest_details", how="left")

# all_hint_hint_.to_csv("../data/all_hint_hint_GMM.csv", index=False)

all_hint_attempt["manifest_details"] = all_hint_attempt["manifest_details"].astype(int)
all_hint_attempt = all_hint_attempt.merge(df_hint_info, on="manifest_details", how="left")
# all_hint_attempt.to_csv("../data/all_hint_attempt_GMM.csv", index=False)


hint_hint["manifest_details"] = hint_hint["manifest_details"].astype(int)
hint_hint = hint_hint.merge(df_hint_info, on="manifest_details", how="left")

# hint_hint.to_csv("../data/hint_hint_GMM.csv", index=False)

hint_attempt["manifest_details"] = hint_attempt["manifest_details"].astype(int)
hint_attempt = hint_attempt.merge(df_hint_info, on="manifest_details", how="left")
# hint_attempt.to_csv("../data/hint_attempt_GMM.csv", index=False)


all_hint_attempt_is_video = all_hint_attempt[all_hint_attempt.is_video == 1]
all_hint_attempt_is_txt = all_hint_attempt[all_hint_attempt.is_video == 0]

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

all_hint_hint_is_video = all_hint_hint_[all_hint_hint_.is_video == 1]
all_hint_hint_is_txt = all_hint_hint_[all_hint_hint_.is_video == 0]
# sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_: " + str(len(all_hint_hint_)))
# sns.distplot(all_hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_is_video: " + str(len(all_hint_hint_is_video)))
# sns.distplot(all_hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_is_txt: " + str(len(all_hint_hint_is_txt)))
# plt.legend()
# plt.title("22. Breaking down all_hint_hint first time by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

hint_hint_is_video = hint_hint[hint_hint.is_video == 1]
hint_hint_is_txt = hint_hint[hint_hint.is_video == 0]
# sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint: " + str(len(hint_hint)))
# sns.distplot(hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_video: " + str(len(hint_hint_is_video)))
# sns.distplot(hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_txt: " + str(len(hint_hint_is_txt)))
# plt.legend()
# plt.title("22.1 Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

# sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint: " + str(len(hint_hint)))
# sns.distplot(hint_hint_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_video: " + str(len(hint_hint_is_video)))
# sns.distplot(hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_txt: " + str(len(hint_hint_is_txt)))
# sns.distplot(hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt: " + str(len(hint_attempt_is_txt)))
# sns.distplot(hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_video: " + str(len(hint_attempt_is_video)))
# plt.legend()
# plt.title("22.2 Breaking down hint_hint first time by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()


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


# hint_type_breakdown(all_hint_attempt_is_video_correct, 2, "23. Means by sklearn [hint-attempt, video, correct]: ",
#                     "23. all_hint_attempt_is_video_correct: ", True)
# hint_type_breakdown(all_hint_attempt_is_video, 3, "24. Means by sklearn [hint-attempt, video]: ",
#                     "24. all_hint_attempt_is_video: ", True)
# hint_type_breakdown(all_hint_attempt_is_video_incorrect, 2, "25. Means by sklearn [hint-attempt, video, incorrect]: ",
#                     "25. all_hint_attempt_is_video_incorrect: ", True)

# sns.distplot(all_hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_video_correct: " + str(len(all_hint_attempt_is_video_correct)), color='g')
# sns.distplot(all_hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_video_incorrect: " + str(len(all_hint_attempt_is_video_incorrect)), color='r')
# sns.distplot(all_hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_video: " + str(len(all_hint_attempt_is_video)))
# plt.legend()
# plt.title("21. all_hint_hint vs all_hint_attempt_video vs all_hint_attempt_is_video_correct vs "
#           "all_hint_attempt_is_video_incorrect")
# plt.show()
#
# hint_type_breakdown(all_hint_attempt_is_txt_correct, 1, "26. Means by sklearn [hint-attempt, text, correct]: ",
#                     "26. all_hint_attempt_is_text_correct: ", True)
# hint_type_breakdown(all_hint_attempt_is_txt, 3, "27. Means by sklearn [hint-attempt, text]: ",
#                     "27. all_hint_attempt_is_text: ", True)
# hint_type_breakdown(all_hint_attempt_is_txt_incorrect, 2, "28. Means by sklearn [hint-attempt, text, incorrect]: ",
#                     "28. all_hint_attempt_is_text_incorrect: ", True)
#
# sns.distplot(all_hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False, label="all_hint_attempt_is_txt_correct: " + str(len(all_hint_attempt_is_txt_correct)),
#              color='g')
# sns.distplot(all_hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_incorrect: " + str(len(all_hint_attempt_is_txt_incorrect)), color='r')
# sns.distplot(all_hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_txt: " + str(len(all_hint_attempt_is_txt)), color='b')
# plt.legend()
# plt.title("20. all_hint_hint vs all_hint_attempt_txt vs all_hint_attempt_txt_correct vs all_hint_attempt_txt_incorrect")
# plt.show()
#
# sns.distplot(hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False, label="hint_attempt_is_txt_correct: " + str(len(hint_attempt_is_txt_correct)),
#              color='g')
# sns.distplot(hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_incorrect: " + str(len(hint_attempt_is_txt_incorrect)), color='r')
# sns.distplot(hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt: " + str(len(hint_attempt_is_txt)), color='b')
# plt.legend()
# plt.title("29. hint_attempt_txt vs hint_attempt_txt_correct vs hint_attempt_txt_incorrect")
# plt.show()
#
# sns.distplot(hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False, label="hint_attempt_is_video_correct: " + str(len(hint_attempt_is_video_correct)),
#              color='g')
# sns.distplot(hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_video_incorrect: " + str(len(hint_attempt_is_video_incorrect)), color='r')
# sns.distplot(hint_attempt_is_video.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_video: " + str(len(hint_attempt_is_video)), color='b')
# plt.legend()
# plt.title("30. hint_attempt_video vs hint_attempt_video_correct vs hint_attempt_video_incorrect")
# plt.show()
#
# sns.distplot(hint_attempt_is_video_correct.log_action_action_pairs_time_taken.values, hist=True, kde=True,
#              rug=False, label="hint_attempt_is_video_correct: " + str(len(hint_attempt_is_video_correct)),
#              color='g')
# sns.distplot(hint_attempt_is_video_incorrect.log_action_action_pairs_time_taken.values, hist=True, kde=True,
#              rug=False,
#              label="hint_attempt_is_video_incorrect: " + str(len(hint_attempt_is_video_incorrect)), color='r')
# plt.legend()
# plt.title("31.1 hint_attempt_video vs hint_attempt_video_correct vs hint_attempt_video_incorrect")
# plt.show()

print("============================================================================================")

quartiles = all_hint_hint_is_txt.hint_body_word_count2.quantile([0.25, 0.5, 0.75])

# q1= 95, q2= 137, q3= 183
all_hint_hint_is_txt_0_q1 = all_hint_hint_is_txt[all_hint_hint_is_txt.hint_body_word_count2 <= quartiles[0.25]]
all_hint_hint_is_txt_q1_q2 = all_hint_hint_is_txt[
    (all_hint_hint_is_txt.hint_body_word_count2 > quartiles[0.25]) & (
                all_hint_hint_is_txt.hint_body_word_count2 <= quartiles[0.5])]
all_hint_hint_is_txt_q2_q3 = all_hint_hint_is_txt[
    (all_hint_hint_is_txt.hint_body_word_count2 > quartiles[0.5]) & (
                all_hint_hint_is_txt.hint_body_word_count2 <= quartiles[0.75])]
all_hint_hint_is_txt_q3_q4 = all_hint_hint_is_txt[(all_hint_hint_is_txt.hint_body_word_count2 > quartiles[0.75])]

# sns.distplot(all_hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_is_txt: " + str(len(all_hint_hint_is_txt)))
# sns.distplot(all_hint_hint_is_txt_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_is_txt_0_q1: " + str(len(all_hint_hint_is_txt_0_q1)))
# sns.distplot(all_hint_hint_is_txt_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_is_txt_q1_q2: " + str(len(all_hint_hint_is_txt_q1_q2)))
# sns.distplot(all_hint_hint_is_txt_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_is_txt_q2_q3: " + str(len(all_hint_hint_is_txt_q2_q3)))
# sns.distplot(all_hint_hint_is_txt_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_hint_is_txt_q3_q4: " + str(len(all_hint_hint_is_txt_q3_q4)))
# plt.legend()
# plt.title("32. Breaking down all_hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

# print("=========================================================================================================")
# print(all_hint_hint_is_txt.hint_body_has_table.value_counts())
# print(all_hint_hint_is_txt_0_q1.hint_body_has_table.value_counts())
# print(all_hint_hint_is_txt_q1_q2.hint_body_has_table.value_counts())
# print(all_hint_hint_is_txt_q2_q3.hint_body_has_table.value_counts())
# print(all_hint_hint_is_txt_q3_q4.hint_body_has_table.value_counts())
# print(all_hint_hint_is_txt_0_q1.hint_body_has_wiris.value_counts())
# print(all_hint_hint_is_txt_q1_q2.hint_body_has_wiris.value_counts())
# print(all_hint_hint_is_txt_q2_q3.hint_body_has_wiris.value_counts())
# print(all_hint_hint_is_txt_q3_q4.hint_body_has_wiris.value_counts())
# print("=========================================================================================================")

quartiles2 = all_hint_hint_is_txt.hint_body_word_count2.quantile([0.25, 0.5, 0.75])

# q1= 95, q2= 134, q3= 191
hint_hint_is_txt_0_q1 = hint_hint_is_txt[hint_hint_is_txt.hint_body_word_count2 <= quartiles2[0.25]]
hint_hint_is_txt_q1_q2 = hint_hint_is_txt[
    (hint_hint_is_txt.hint_body_word_count2 > quartiles2[0.25]) & (
                hint_hint_is_txt.hint_body_word_count2 <= quartiles2[0.5])]
hint_hint_is_txt_q2_q3 = hint_hint_is_txt[
    (hint_hint_is_txt.hint_body_word_count2 > quartiles2[0.5]) & (
                hint_hint_is_txt.hint_body_word_count2 <= quartiles2[0.75])]
hint_hint_is_txt_q3_q4 = hint_hint_is_txt[(hint_hint_is_txt.hint_body_word_count2 > quartiles2[0.75])]
#
# sns.distplot(hint_hint_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_txt: " + str(len(hint_hint_is_txt)))
# sns.distplot(hint_hint_is_txt_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_txt_0_q1: " + str(len(hint_hint_is_txt_0_q1)))
# sns.distplot(hint_hint_is_txt_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_txt_q1_q2: " + str(len(hint_hint_is_txt_q1_q2)))
# sns.distplot(hint_hint_is_txt_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_txt_q2_q3: " + str(len(hint_hint_is_txt_q2_q3)))
# sns.distplot(hint_hint_is_txt_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_hint_is_txt_q3_q4: " + str(len(hint_hint_is_txt_q3_q4)))
# plt.legend()
# plt.title("32.1 Breaking down hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

# print("=========================================================================================================")
# print(hint_hint_is_txt.hint_body_has_table.value_counts())
# print(hint_hint_is_txt_0_q1.hint_body_has_table.value_counts())
# print(hint_hint_is_txt_q1_q2.hint_body_has_table.value_counts())
# print(hint_hint_is_txt_q2_q3.hint_body_has_table.value_counts())
# print(hint_hint_is_txt_q3_q4.hint_body_has_table.value_counts())
# print(hint_hint_is_txt_0_q1.hint_body_has_wiris.value_counts())
# print(hint_hint_is_txt_q1_q2.hint_body_has_wiris.value_counts())
# print(hint_hint_is_txt_q2_q3.hint_body_has_wiris.value_counts())
# print(hint_hint_is_txt_q3_q4.hint_body_has_wiris.value_counts())
# print("=========================================================================================================")

# print("============================================================================================")
#
# quartiles = all_hint_attempt_is_txt.hint_body_word_count2.quantile([0.25, 0.5, 0.75])
#
# print(quartiles)
print("============================================================================================")

# q1= 95, q2= 137, q3= 183
all_hint_attempt_is_txt_0_q1 = all_hint_attempt_is_txt[all_hint_attempt_is_txt.hint_body_word_count2 <= quartiles[0.25]]
all_hint_attempt_is_txt_q1_q2 = all_hint_attempt_is_txt[
    (all_hint_attempt_is_txt.hint_body_word_count2 > quartiles[0.25]) & (
                all_hint_attempt_is_txt.hint_body_word_count2 <= quartiles[0.5])]
all_hint_attempt_is_txt_q2_q3 = all_hint_attempt_is_txt[
    (all_hint_attempt_is_txt.hint_body_word_count2 > quartiles[0.5]) & (
                all_hint_attempt_is_txt.hint_body_word_count2 <= quartiles[0.75])]
all_hint_attempt_is_txt_q3_q4 = all_hint_attempt_is_txt[
    (all_hint_attempt_is_txt.hint_body_word_count2 > quartiles[0.75])]

# sns.distplot(all_hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_txt: " + str(len(all_hint_attempt_is_txt)))
# sns.distplot(all_hint_attempt_is_txt_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_txt_0_q1: " + str(len(all_hint_attempt_is_txt_0_q1)))
# sns.distplot(all_hint_attempt_is_txt_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_txt_q1_q2: " + str(len(all_hint_attempt_is_txt_q1_q2)))
# sns.distplot(all_hint_attempt_is_txt_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_txt_q2_q3: " + str(len(all_hint_attempt_is_txt_q2_q3)))
# sns.distplot(all_hint_attempt_is_txt_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_txt_q3_q4: " + str(len(all_hint_attempt_is_txt_q3_q4)))
# plt.legend()
# plt.title("33. Breaking down all_hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

# print("=========================================================================================================")
# print(all_hint_attempt_is_txt.hint_body_has_table.value_counts())
# print(all_hint_attempt_is_txt_0_q1.hint_body_has_table.value_counts())
# print(all_hint_attempt_is_txt_q1_q2.hint_body_has_table.value_counts())
# print(all_hint_attempt_is_txt_q2_q3.hint_body_has_table.value_counts())
# print(all_hint_attempt_is_txt_q3_q4.hint_body_has_table.value_counts())
# print(all_hint_attempt_is_txt_0_q1.hint_body_has_wiris.value_counts())
# print(all_hint_attempt_is_txt_q1_q2.hint_body_has_wiris.value_counts())
# print(all_hint_attempt_is_txt_q2_q3.hint_body_has_wiris.value_counts())
# print(all_hint_attempt_is_txt_q3_q4.hint_body_has_wiris.value_counts())
# print("=========================================================================================================")

quartiles2 = hint_attempt_is_txt.hint_body_word_count2.quantile([0.25, 0.5, 0.75])

# q1= 95, q2= 134, q3= 191
hint_attempt_is_txt_0_q1 = hint_attempt_is_txt[hint_attempt_is_txt.hint_body_word_count2 <= quartiles2[0.25]]
hint_attempt_is_txt_q1_q2 = hint_attempt_is_txt[
    (hint_attempt_is_txt.hint_body_word_count2 > quartiles2[0.25]) & (
                hint_attempt_is_txt.hint_body_word_count2 <= quartiles2[0.5])]
hint_attempt_is_txt_q2_q3 = hint_attempt_is_txt[
    (hint_attempt_is_txt.hint_body_word_count2 > quartiles2[0.5]) & (
                hint_attempt_is_txt.hint_body_word_count2 <= quartiles2[0.75])]
hint_attempt_is_txt_q3_q4 = hint_attempt_is_txt[(hint_attempt_is_txt.hint_body_word_count2 > quartiles2[0.75])]

# sns.distplot(hint_attempt_is_txt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt: " + str(len(hint_attempt_is_txt)))
# sns.distplot(hint_attempt_is_txt_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_0_q1: " + str(len(hint_attempt_is_txt_0_q1)))
# sns.distplot(hint_attempt_is_txt_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q1_q2: " + str(len(hint_attempt_is_txt_q1_q2)))
# sns.distplot(hint_attempt_is_txt_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q2_q3: " + str(len(hint_attempt_is_txt_q2_q3)))
# sns.distplot(hint_attempt_is_txt_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q3_q4: " + str(len(hint_attempt_is_txt_q3_q4)))
# plt.legend()
# plt.title("33.1 Breaking down hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

print("=========================================================================================================")

q0_q1_corrrect = hint_attempt_is_txt_0_q1[hint_attempt_is_txt_0_q1.pr_answered_correctly_pair == 1]
q0_q1_incorrrect = hint_attempt_is_txt_0_q1[hint_attempt_is_txt_0_q1.pr_answered_correctly_pair == 0]

# sns.distplot(hint_attempt_is_txt_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_0_q1: " + str(len(hint_attempt_is_txt_0_q1)))
# sns.distplot(q0_q1_corrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q0_q1_correct: " + str(len(q0_q1_corrrect)))
# sns.distplot(q0_q1_incorrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q0_q1_incorrect: " + str(len(q0_q1_incorrrect)))
# plt.legend()
# plt.title("33.2 hint attempt q0_q1 correct vs incorrect")
# plt.show()

q1_q2_corrrect = hint_attempt_is_txt_q1_q2[hint_attempt_is_txt_q1_q2.pr_answered_correctly_pair == 1]
q1_q2_incorrrect = hint_attempt_is_txt_q1_q2[hint_attempt_is_txt_q1_q2.pr_answered_correctly_pair == 0]

# sns.distplot(hint_attempt_is_txt_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q1_q2: " + str(len(hint_attempt_is_txt_q1_q2)))
# sns.distplot(q0_q1_corrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q1_q2_correct: " + str(len(q1_q2_corrrect)))
# sns.distplot(q0_q1_incorrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q1_q2_incorrect: " + str(len(q1_q2_incorrrect)))
# plt.legend()
# plt.title("33.3 hint attempt q1_q2 correct vs incorrect")
# plt.show()

q2_q3_corrrect = hint_attempt_is_txt_q2_q3[hint_attempt_is_txt_q2_q3.pr_answered_correctly_pair == 1]
q2_q3_incorrrect = hint_attempt_is_txt_q2_q3[hint_attempt_is_txt_q2_q3.pr_answered_correctly_pair == 0]

# sns.distplot(hint_attempt_is_txt_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q2_q3: " + str(len(hint_attempt_is_txt_q2_q3)))
# sns.distplot(q2_q3_corrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q2_q3_correct: " + str(len(q2_q3_corrrect)))
# sns.distplot(q2_q3_incorrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q2_q3_incorrect: " + str(len(q2_q3_incorrrect)))
# plt.legend()
# plt.title("33.4 hint attempt q2_q3 correct vs incorrect")
# plt.show()

# q3_q4_corrrect = hint_attempt_is_txt_q3_q4[hint_attempt_is_txt_q3_q4.pr_answered_correctly_pair == 1]
# q3_q4_incorrrect = hint_attempt_is_txt_q3_q4[hint_attempt_is_txt_q3_q4.pr_answered_correctly_pair == 0]
#
# sns.distplot(hint_attempt_is_txt_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q3_q4: " + str(len(hint_attempt_is_txt_q3_q4)))
# sns.distplot(q3_q4_corrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q3_q4_correct: " + str(len(q3_q4_corrrect)))
# sns.distplot(q3_q4_incorrrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_q3_q4_incorrect: " + str(len(q3_q4_incorrrect)))
# plt.legend()
# plt.title("33.5 hint attempt q3_q4 correct vs incorrect")
# plt.show()

# print("=========================================================================================================")
# print(hint_attempt_is_txt.hint_body_has_table.value_counts())
# print(hint_attempt_is_txt_0_q1.hint_body_has_table.value_counts())
# print(hint_attempt_is_txt_q1_q2.hint_body_has_table.value_counts())
# print(hint_attempt_is_txt_q2_q3.hint_body_has_table.value_counts())
# print(hint_attempt_is_txt_q3_q4.hint_body_has_table.value_counts())
# print(hint_attempt_is_txt_0_q1.hint_body_has_wiris.value_counts())
# print(hint_attempt_is_txt_q1_q2.hint_body_has_wiris.value_counts())
# print(hint_attempt_is_txt_q2_q3.hint_body_has_wiris.value_counts())
# print(hint_attempt_is_txt_q3_q4.hint_body_has_wiris.value_counts())
# print("=========================================================================================================")

quartiles2 = hint_attempt_is_txt_correct.hint_body_word_count2.quantile([0.25, 0.5, 0.75])

# q1= 95, q2= 134, q3= 191
hint_attempt_is_txt_correct_0_q1 = hint_attempt_is_txt_correct[
    hint_attempt_is_txt_correct.hint_body_word_count2 <= quartiles2[0.25]]
hint_attempt_is_txt_correct_q1_q2 = hint_attempt_is_txt_correct[
    (hint_attempt_is_txt_correct.hint_body_word_count2 > quartiles2[0.25]) & (
                hint_attempt_is_txt_correct.hint_body_word_count2 <= quartiles2[0.5])]
hint_attempt_is_txt_correct_q2_q3 = hint_attempt_is_txt_correct[
    (hint_attempt_is_txt_correct.hint_body_word_count2 > quartiles2[0.5]) & (
                hint_attempt_is_txt_correct.hint_body_word_count2 <= quartiles2[0.75])]
hint_attempt_is_txt_correct_q3_q4 = hint_attempt_is_txt_correct[
    (hint_attempt_is_txt_correct.hint_body_word_count2 > quartiles2[0.75])]

# sns.distplot(hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_correct: " + str(len(hint_attempt_is_txt_correct)))
# sns.distplot(hint_attempt_is_txt_correct_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_correct_0_q1: " + str(len(hint_attempt_is_txt_correct_0_q1)))
# sns.distplot(hint_attempt_is_txt_correct_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_correct_q1_q2: " + str(len(hint_attempt_is_txt_correct_q1_q2)))
# sns.distplot(hint_attempt_is_txt_correct_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_correct_q2_q3: " + str(len(hint_attempt_is_txt_correct_q2_q3)))
# sns.distplot(hint_attempt_is_txt_correct_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_correct_q3_q4: " + str(len(hint_attempt_is_txt_correct_q3_q4)))
# plt.legend()
# plt.title("33.2 Breaking down hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

quartiles2 = all_hint_attempt_is_txt_correct.hint_body_word_count2.quantile([0.25, 0.5, 0.75])

# q1= 95, q2= 134, q3= 191
all_hint_attempt_is_txt_correct_0_q1 = all_hint_attempt_is_txt_correct[
    all_hint_attempt_is_txt_correct.hint_body_word_count2 <= quartiles2[0.25]]
all_hint_attempt_is_txt_correct_q1_q2 = all_hint_attempt_is_txt_correct[
    (all_hint_attempt_is_txt_correct.hint_body_word_count2 > quartiles2[0.25]) & (
                all_hint_attempt_is_txt_correct.hint_body_word_count2 <= quartiles2[0.5])]
all_hint_attempt_is_txt_correct_q2_q3 = all_hint_attempt_is_txt_correct[
    (all_hint_attempt_is_txt_correct.hint_body_word_count2 > quartiles2[0.5]) & (
                all_hint_attempt_is_txt_correct.hint_body_word_count2 <= quartiles2[0.75])]
all_hint_attempt_is_txt_correct_q3_q4 = all_hint_attempt_is_txt_correct[
    (all_hint_attempt_is_txt_correct.hint_body_word_count2 > quartiles2[0.75])]

# sns.distplot(all_hint_attempt_is_txt_correct.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="all_hint_attempt_is_txt_correct: " + str(len(all_hint_attempt_is_txt_correct)))
# sns.distplot(all_hint_attempt_is_txt_correct_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_correct_0_q1: " + str(len(all_hint_attempt_is_txt_correct_0_q1)))
# sns.distplot(all_hint_attempt_is_txt_correct_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_correct_q1_q2: " + str(len(all_hint_attempt_is_txt_correct_q1_q2)))
# sns.distplot(all_hint_attempt_is_txt_correct_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_correct_q2_q3: " + str(len(all_hint_attempt_is_txt_correct_q2_q3)))
# sns.distplot(all_hint_attempt_is_txt_correct_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_correct_q3_q4: " + str(len(all_hint_attempt_is_txt_correct_q3_q4)))
# plt.legend()
# plt.title("33.3 Breaking down hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

quartiles2 = hint_attempt_is_txt_incorrect.hint_body_word_count2.quantile([0.25, 0.5, 0.75])

# q1= 95, q2= 134, q3= 191
hint_attempt_is_txt_incorrect_0_q1 = hint_attempt_is_txt_incorrect[
    hint_attempt_is_txt_incorrect.hint_body_word_count2 <= quartiles2[0.25]]
hint_attempt_is_txt_incorrect_q1_q2 = hint_attempt_is_txt_incorrect[
    (hint_attempt_is_txt_incorrect.hint_body_word_count2 > quartiles2[0.25]) & (
                hint_attempt_is_txt_incorrect.hint_body_word_count2 <= quartiles2[0.5])]
hint_attempt_is_txt_incorrect_q2_q3 = hint_attempt_is_txt_incorrect[
    (hint_attempt_is_txt_incorrect.hint_body_word_count2 > quartiles2[0.5]) & (
                hint_attempt_is_txt_incorrect.hint_body_word_count2 <= quartiles2[0.75])]
hint_attempt_is_txt_incorrect_q3_q4 = hint_attempt_is_txt_incorrect[
    (hint_attempt_is_txt_incorrect.hint_body_word_count2 > quartiles2[0.75])]

# sns.distplot(hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
#              label="hint_attempt_is_txt_incorrect: " + str(len(hint_attempt_is_txt_incorrect)))
# sns.distplot(hint_attempt_is_txt_incorrect_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_incorrect_0_q1: " + str(len(hint_attempt_is_txt_incorrect_0_q1)))
# sns.distplot(hint_attempt_is_txt_incorrect_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_incorrect_q1_q2: " + str(len(hint_attempt_is_txt_incorrect_q1_q2)))
# sns.distplot(hint_attempt_is_txt_incorrect_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_incorrect_q2_q3: " + str(len(hint_attempt_is_txt_incorrect_q2_q3)))
# sns.distplot(hint_attempt_is_txt_incorrect_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="hint_attempt_is_txt_incorrect_q3_q4: " + str(len(hint_attempt_is_txt_incorrect_q3_q4)))
# plt.legend()
# plt.title("33.4 Breaking down hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()

quartiles2 = all_hint_attempt_is_txt_incorrect.hint_body_word_count2.quantile([0.25, 0.5, 0.75])

# q1= 95, q2= 134, q3= 191
all_hint_attempt_is_txt_incorrect_0_q1 = all_hint_attempt_is_txt_incorrect[
    all_hint_attempt_is_txt_incorrect.hint_body_word_count2 <= quartiles2[0.25]]
all_hint_attempt_is_txt_incorrect_q1_q2 = all_hint_attempt_is_txt_incorrect[
    (all_hint_attempt_is_txt_incorrect.hint_body_word_count2 > quartiles2[0.25]) & (
                all_hint_attempt_is_txt_incorrect.hint_body_word_count2 <= quartiles2[0.5])]
all_hint_attempt_is_txt_incorrect_q2_q3 = all_hint_attempt_is_txt_incorrect[
    (all_hint_attempt_is_txt_incorrect.hint_body_word_count2 > quartiles2[0.5]) & (
                all_hint_attempt_is_txt_incorrect.hint_body_word_count2 <= quartiles2[0.75])]
all_hint_attempt_is_txt_incorrect_q3_q4 = all_hint_attempt_is_txt_incorrect[
    (all_hint_attempt_is_txt_incorrect.hint_body_word_count2 > quartiles2[0.75])]

# sns.distplot(all_hint_attempt_is_txt_incorrect.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_incorrect: " + str(len(all_hint_attempt_is_txt_incorrect)),
#              kde_kws={"linestyle": "--"})
# sns.distplot(all_hint_attempt_is_txt_incorrect_0_q1.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_incorrect_0_q1: " + str(len(all_hint_attempt_is_txt_incorrect_0_q1)),
#              kde_kws={"linestyle": "-."})
# sns.distplot(all_hint_attempt_is_txt_incorrect_q1_q2.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_incorrect_q1_q2: " + str(len(all_hint_attempt_is_txt_incorrect_q1_q2)),
#              kde_kws={"linestyle": ":"})
# sns.distplot(all_hint_attempt_is_txt_incorrect_q2_q3.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_incorrect_q2_q3: " + str(len(all_hint_attempt_is_txt_incorrect_q2_q3)),
#              kde_kws={"linestyle": "-"})
# sns.distplot(all_hint_attempt_is_txt_incorrect_q3_q4.log_action_action_pairs_time_taken.values, hist=False, kde=True,
#              rug=False,
#              label="all_hint_attempt_is_txt_incorrect_q3_q4: " + str(len(all_hint_attempt_is_txt_incorrect_q3_q4)))
# plt.legend()
# plt.title("33.5 Breaking down hint_hint with length of the hint by incorporating attempts[depth: 2] as well in \n"
#           "actual time > 1.5, z-score[-3,3] vs hint_hint with video vs txt")
# plt.show()


sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="( Hint Request, Hint Request)")
sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="( Hint Request, Attempt)", kde_kws={"linestyle": "--"})
plt.xlabel("log transformed action pair response time")
plt.show()

sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="( Hint Request, Hint Request)")
sns.distplot(hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="( Hint Request, Attempt)", kde_kws={"linestyle": "--"})
plt.xlabel("log transformed action pair response time")
plt.show()

sns.set(context="talk", style="whitegrid", font_scale=1.7, rc={'figure.figsize': (30, 20)})

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

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
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

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
plt.show()

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
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(all_hint_hint_.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Hint Request)", ax=ax1)
sns.distplot(all_hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Attempt)", kde_kws={"linestyle": "--"}, ax=ax1)

sns.distplot(hint_hint.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Hint Request)", ax=ax2)
sns.distplot(hint_attempt.log_action_action_pairs_time_taken.values, hist=False, kde=True, rug=False,
             label="(Hint Request, Attempt)", kde_kws={"linestyle": "--"}, ax=ax2)
ax1.set_title("1. action pair for all Hint Request")
ax1.set_xlabel("log transformed action pair response time")

ax2.set_title("2. action pair for first Hint Request")
ax2.set_xlabel("log transformed action pair response time")
plt.legend()
# plt.xlabel("log transformed action pair response time")
plt.show()
