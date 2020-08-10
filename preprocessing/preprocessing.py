import pandas as pd

main_df = pd.read_csv('../data/action_logs_19_20_random_sample_20k.csv')

main_df.describe()

main_df['ps'] = (main_df.path.astype(str) + "/").str.extract("\/(.*?)\/")
main_df['pr'] = (main_df.path.astype(str) + "/").str.extract("\/PR(.*?)\#")
main_df = main_df[main_df.ps != '']
main_df.pr.fillna(-1, inplace=True)

# sort the data in the following order; just being cautious to ensure order
# ps > assignment_log_id > users > pr > action_unix_time
main_df.sort_values(by=['ps', 'assignment_log_id', 'user_xid', 'pr', 'action_unix_time'], inplace=True)
main_df.reset_index(drop=True)

# because the analysis is from one action to the next
# let us combine each action type with the next action type per user > ps > pr -> "action_action_pairs"
# we also calculate the time spend between the two action pairs by the user -> "action_action_pairs_time_taken"
# new features that generated for clarity
main_df['action_action'] = 'test1'
main_df['action_action_pairs'] = 'test'
main_df['action_action_pairs_time_taken'] = 0


# this method basically calculates the action sequence of students in a ps into action pairs to determine which
# action took the most time
def arrange_action_pairs(_user_xid):
    test_df = main_df.loc[main_df.user_xid == _user_xid]
    # find all the unique PRs the user has attempted in the PS
    prs = test_df.pr.unique()
    # print(prs)

    # NOTE: let us decompose the actions of a user on the PS per problem by combining the actions into pairs
    #  "action_action_pairs" and calculate the time taken by the user between the two actions
    #  "action_action_pairs_time_taken" should we take into account user selected continue action? YES: because how
    #  much time a student takes to start next problem might be indicative of how motivated they are to tackle the PS
    #  NO: because the information isn't actionable in a productive way. If we do manage to have some insightful
    #  findings in the favor of short time difference between problem completion and continue action then Forcing
    #  people to start the next problem might trigger discontent due to loss of autonomy
    # RESEARCH QUESTION: This might be an interesting research question though. Shelving it for now will comeback to
    #  it later.
    # NOTE: keep in mind this work is very similar to what Anthony had explored last year[2019] as "Refusal" to
    #  answer we might reformat it to how much or what nature of refusal is better: it might be the case that within in
    #  day refusal has a superior conversion rate to next day conversion rate when they eventually try to answer the
    #  problem they displayed refusal on.

    # To do this modify the sort_values at the beginning of the code order by removing 'pr' and there would be some
    # finiking required with the "action_type : UserSelectedContinueAction" to match them to their corresponding pr ids
    for pr in prs:
        if pr != -1:
            indices = test_df.loc[test_df.pr == pr].index.to_list()
            indices_length = len(indices)
            for i in range(indices_length):
                df_index = indices[i]
                main_df.loc[[df_index], ['action_action']] = "testing : " + str(df_index)
                if i < (indices_length - 1):
                    df_index_next = indices[i + 1]
                    main_df.loc[[df_index], ['action_action_pairs']] = main_df.loc[[df_index]].action_type.str.cat(
                        main_df.loc[[df_index_next]].action_type.values.astype(str), sep="_").values[0]
                    main_df.loc[[df_index], ['action_action_pairs_time_taken']] = \
                    main_df.loc[[df_index_next]]['action_unix_time'].values[0] - \
                    main_df.loc[[df_index]]['action_unix_time'].values[0]
                else:
                    main_df.loc[[df_index], ['action_action_pairs']] = main_df.loc[[df_index]].action_type.values[0]
                    main_df.loc[[df_index], ['action_action_pairs_time_taken']] = 0

            # if (indices_length - 1) > 1:
            #     print("============================================================")
            #     print("action pairs", indices_length)
            #     # print(main_df.loc[[indices[1]]])
            #     # print(main_df.loc[[indices[1]]]['action_action_pairs'])
            #     print(main_df.loc[[indices[1]]].action_type.str.cat(
            #         main_df.loc[[indices[1]]].action_type.values.astype(str), sep="_").values[0])
            #     print("action pairs time", indices[1])
            #     # print(main_df.loc[[indices[1]]]['action_action_pairs_time_taken'])
            #     print(main_df.loc[[indices[1 + 1]]]['action_unix_time'].values[0] -
            #           main_df.loc[[indices[1]]]['action_unix_time'].values[0])
            #     print("============================================================")


# the problem set PS5919713 seems to have the most records 4412
# the ps == "PS5919713" drops the assignment started action as path value is '/' only
# print(main_df.ps.value_counts())
# NOTE: the problem set PS1065570 seems to be a SkillBuilder
# TODO: talk to Anthony about this
pss = main_df.ps.unique()
print(main_df.ps.value_counts())
for ps in pss:
    if ps != "":
        user_xids = main_df.loc[main_df.ps == ps].user_xid.unique()
        print(ps, len(user_xids))
        for user_xid in user_xids:
            arrange_action_pairs(user_xid)


main_df.to_csv("../data/RTD_preprocessed_new.csv", index=False)
# main_df = main_df.loc[main_df.ps == "PS5919713"]
# user_ids = main_df.user_xid.unique()
#
# for user_xid in user_ids:
#     arrange_action_pairs(user_xid)

# a total of 72 assignments were assigned using this problem set
# 65964 has 49 students all others have very few students [1,4]
# this of course is an inconvenience because it means the assignment to user ratio is not what we wanted
# assignment_ids = main_df.assignment_xid.unique()
# for assignment_xid in assignment_ids:
#     print(assignment_xid, "  :  ", main_df.loc[main_df.assignment_xid == assignment_xid].user_xid.unique().shape)
