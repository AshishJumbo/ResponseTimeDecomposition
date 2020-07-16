import pandas as pd

main_df = pd.read_csv('../data/action_logs_19_20_random_sample_20k.csv')

main_df.describe()

main_df['ps'] = (main_df.path.astype(str) + "/").str.extract("\/(.*?)\/")
main_df['pr'] = (main_df.path.astype(str) + "/").str.extract("\/PR(.*?)\#")
main_df = main_df[main_df.ps != '']
main_df.pr.fillna(-1, inplace=True)

# sort the data in the following order; just being cautious to ensure order
# ps > assignment_log_id > users > pr > action_unix_time

# lose the pr as prs in ps might be in different order and new problems have larger pr even if it was the first problem in ps
# main_df.sort_values(by=['ps', 'assignment_log_id', 'user_xid', 'pr', 'action_unix_time'], inplace=True)
# NOTE: removed the pr as a consequence to this ^^^ concern
main_df.sort_values(by=['ps', 'assignment_log_id', 'user_xid', 'action_unix_time'], inplace=True)
main_df.reset_index(drop=True)

# because the analysis is from one action to the next
# let us combine each action type with the next action type per user > ps > pr -> "action_action_pairs"
# we also calculate the time spend between the two action pairs by the user -> "action_action_pairs_time_taken"
# new features that generated for clarity
main_df['action_action_pairs'] = 'test'
main_df['action_action_pairs_time_taken'] = 0
main_df['pr_answered_correctly_pair'] = 0
main_df['pr_answer_attempts_by_user'] = 0
main_df['pr_hints_requested_by_user'] = 0
# main_df.to_csv("../data/RTD_data_randomsample_20K_new.csv", index=False)

# once the RTD_data_randomsample_20K is exported use the instructions in libreoffice.md to generate the feature
# values; had to do this because it was taking too long to generate the feature values using python as Python by
# default is a single threaded operation i.e. it uses single core
