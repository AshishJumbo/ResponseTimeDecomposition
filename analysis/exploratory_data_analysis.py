import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_main = pd.read_csv('../data/RTD_data_randomsample_20K.csv')

print(df_main.describe())

# PS level action don't have a PR associated with them
df_main = df_main[df_main.pr != -1]

# the ProblemFinishedAction info is accounted for by the StudentResponseAction_ProblemFinishedAction pair so dropping
# them as well The StudentResponseAction_ProblemFinishedAction action itself is a automated action sequence pair
# probably should be dropping that as well

df_main = df_main[df_main.action_action_pairs != "ProblemFinishedAction"]
# need more clarity on this because the ProblemResumedAction_StudentResponseAction action pair alone doesn't convey
# if the student response was correct
#
# TODO: talk to Anthony about this do we need to add score as a feature as well or is there a neater workaround
#  because StudentResponseAction_ProblemFinishedAction indicates a correct response but not full correct response
#  on the first try [Answered]
df_main = df_main[df_main.action_action_pairs != "StudentResponseAction_ProblemFinishedAction"]

# a lot of automated action sequence pairs that are generated from code take less than 1 second  so dropping those
# the total action pairs have been brought down to 43 pairs
df_main = df_main[df_main.action_action_pairs_time_taken > 1]

# winsorizing the data at 24 hours
df_main.action_action_pairs_time_taken.clip(None, 86400, inplace=True)
df_main["log_action_action_pair_time"] = np.log(df_main.action_action_pairs_time_taken)


# I still have some concerns about certain pairs
# 1. ProblemResumedAction_ProblemResumedAction
# 2. ProblemStartedAction_ProblemResumedAction
# these actions have the student disengaging from the system for sometime
# so should we assume that the students' mental model to solve a problem
# will already have been reset when they resume
# NOTE: This could also be an interesting research question
# RESEARCH QUESTION: How much time needs to pass before a students' mental model loses contextual awarenees
#  about solving a problem? IS there a relation between ..._ProblemResumedAction time and
#  ProblemResumedAction_StudentResponseAction
#  maybe once we z-score the problems difficulty we should compare the users time when problem was
#  resumed vs when they answered similar difficulty problem to check if there is drastic difference in
#  the time using t-test[Would it be paired or two sample]
#  Anthony: can we caracterise student resume action into various categories and hopothesise why the student left?
#  within day and within hour would be interesting scenarios
# i.e. can we drop these and only analyse the ProblemResumedAction_StudentResponseAction and consider
# it to be very similar to ProblemStartedAction_StudentResponseAction

# df_plt = df_main[df_main.action_action_pairs == 'StudentResponseAction_StudentResponseAction']
# df_plt = df_main[df_main.action_action_pairs == 'ProblemStartedAction_StudentResponseAction']


def plotCharts(df, fig_width, fig_height, title):
    sns.set(style="whitegrid")
    sns.set(rc={'figure.figsize': (fig_width, fig_height)})
    # sns.set(rc={"figure.dpi": 200})

    ax = sns.boxplot(x="action_action_pairs_time_taken", y="action_action_pairs", data=df)

    # Calculate number of obs per group & median to position labels
    # labels = [label.get_text() for label in ax.get_yticklabels()]
    # medians = df.groupby(['action_action_pairs'])['action_action_pairs_time_taken'].median().values
    # value_counts = df['action_action_pairs'].value_counts()
    # value_counts = value_counts.reindex(index=labels)
    # # nobs = df['action_action_pairs'].value_counts().values
    # # nobs2 = df['action_action_pairs'].value_counts().index.tolist()
    # nobs = value_counts.values
    # nobs2 = value_counts.index.tolist()
    # print(title)
    # print(value_counts)
    # print('====================================================')
    # nobs = [str(x) for x in nobs.tolist()]
    # nobs = ["n: " + i for i in nobs]
    #
    # for tick in range(len(nobs)):
    #     # idx = nobs2.index(labels[tick].get_text())
    #
    #     print(labels[tick].get_text(), medians[tick], len(nobs) - tick - 1, nobs[tick])
    #     ax.text(medians[tick], len(nobs) - tick - 1, nobs[tick],
    #             horizontalalignment='center', size='x-small', color='b', weight='semibold', bbox=dict(boxstyle="round",
    #                                                                                                   ec=(1., 0.5, 0.5),
    #                                                                                                   fc=(1., 0.8, 0.8),
    #                                                                                                   ))

    # sns.plt.show()
    plt.title("box-plot: " + title)
    plt.show()

    sns.distplot(df.log_action_action_pair_time.values, kde=True, rug=False)
    plt.title("histogram:" + title)
    plt.show()



plotCharts(df_main, 40, 20, "Exploring data RTD at a PR level[time: 24 hrs]")

# let's just check for pairs that occur more than a 100 times
action_action_greater_100 = df_main.action_action_pairs.value_counts()
action_action_greater_100 = action_action_greater_100[action_action_greater_100 > 100]
action_action_greater_100_index = action_action_greater_100.index

df_main = df_main[df_main.action_action_pairs.isin(action_action_greater_100_index)]
plotCharts(df_main, 40, 20, "Dropping all action_action pairs that occur less than 100 times [time: 24 hrs]")

# sns.boxplot(x="action_action_pairs_time_taken", y="action_action_pairs", data=df_main)
# plt.show()

# _ProblemResumedAction is taking too much time
action_action_greater_100_index = action_action_greater_100_index[
    ~action_action_greater_100_index.str.contains("_ProblemResumedAction")]

df_main.action_action_pairs_time_taken.clip(None, 1800, inplace=True)

df_main = df_main[df_main.action_action_pairs.isin(action_action_greater_100_index)]
plotCharts(df_main, 40, 20, "Dropping all pairs that end in _ProblemResumedAction and winsorizing the records to 30 "
                            "mins")

df_main["log_action_action_pair_time"] = np.log(df_main.action_action_pairs_time_taken)

df_main.action_action_pairs_time_taken.clip(None, 600, inplace=True)
plotCharts(df_main, 40, 20, " winsorizing the records to 10 mins")

# df_main = df_main[df_main.action_action_pairs.isin(action_action_greater_100_index)]
# sns.boxplot(x="action_action_pairs_time_taken", y="action_action_pairs", data=df_main)
# plt.show()
