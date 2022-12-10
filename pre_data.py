import pickle

import pandas as pd


def pre_nyc():
    file = './data/nyc/nyc_dataset.csv'
    tv_num = 9989  # item_num
    user_id = 0
    month = 4
    day = 3
    history = []
    timeslot_1 = []  # itemset in dawn/deep
    timeslot_2 = []  # itemset in morning
    timeslot_3 = []  # itemset in afternoon
    timeslot_4 = []  # itemset in evening
    line_count = 0

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[1:]
        line_num = len(lines)
        for line in lines:
            line_count += 1
            line = line.strip().split(',')
            date = line[3].strip().split('-')
            hour = int(line[4])
            tv_id = int(line[2])

            # split time segment data and generate one period sample for one user
            if (int(date[0]) != month or int(date[1]) != day) or (line_count == line_num) or (int(line[1]) != user_id):
                history_day_user = []
                history_day_user.append(user_id)

                if int(line[1]) != user_id:
                    user_id = int(line[1])

                if timeslot_4:
                    deep_night = list(set(timeslot_4))
                else:
                    deep_night = [tv_num]
                history_day_user.append(deep_night)

                if timeslot_1:
                    moring = list(set(timeslot_1))
                else:
                    moring = [tv_num]
                history_day_user.append(moring)

                if timeslot_2:
                    afternoon = list(set(timeslot_2))
                else:
                    afternoon = [tv_num]
                history_day_user.append(afternoon)

                if timeslot_3:
                    evening = list(set(timeslot_3))
                else:
                    evening = [tv_num]
                history_day_user.append(evening)

                history_day_user = tuple(history_day_user)  # 转换成元祖的形式
                history.append(history_day_user)

                month = int(date[0])
                day = int(date[1])
                timeslot_1 = []
                timeslot_2 = []
                timeslot_3 = []
                timeslot_4 = []

            # morining
            if hour >= 7 and hour <= 11:
                timeslot_1.append(tv_id)

            # afternoon
            if hour >= 12 and hour <= 18:
                timeslot_2.append(tv_id)

            # evening
            if hour >= 19 and hour <= 22:
                timeslot_3.append(tv_id)

            # dawn
            if (hour >= 0 and hour <= 6) or (hour == 23):
                timeslot_4.append(tv_id)
    f.close()

    # get consective period sample for one user
    new_history = []
    for i in range(len(history) - 1):
        tuple_pre = history[i]  # last period
        tuple_nex = history[i + 1]  # current period
        user_id_pre = tuple_pre[0]
        user_id_next = tuple_nex[0]

        if user_id_pre == user_id_next:
            next_mor_bas, next_aft_bas, next_eve_bas, next_deep_bas = tuple_nex[1:]
            user_tuple_pre = list(tuple_pre)
            user_tuple_pre.append(next_mor_bas)
            user_tuple_pre.append(next_aft_bas)
            user_tuple_pre.append(next_eve_bas)
            user_tuple_pre.append(next_deep_bas)
            new_history.append(tuple(user_tuple_pre))

    f = open('./data/nyc/nyc_for_our.pkl', 'wb')
    pickle.dump(new_history, f)
    f.close()


def pre_tky():
    file = './data/tky/tky_dataset.csv'
    tv_num = 15177  # item_num
    user_id = 0
    month = 4
    day = 7
    history = []
    timeslot_1 = []  # itemset in dawn/deep
    timeslot_2 = []  # itemset in morning
    timeslot_3 = []  # itemset in afternoon
    timeslot_4 = []  # itemset in evening
    line_count = 0

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[1:]
        line_num = len(lines)
        for line in lines:
            line_count += 1
            line = line.strip().split(',')
            date = line[3].strip().split('-')
            hour = int(line[4])
            tv_id = int(line[1])

            # split time segment data and generate one period sample for one user
            if (int(date[0]) != month or int(date[1]) != day) or (line_count == line_num) or (int(line[0]) != user_id):
                history_day_user = []
                history_day_user.append(user_id)

                if int(line[0]) != user_id:
                    user_id = int(line[0])

                if timeslot_4:
                    deep_night = list(set(timeslot_4))
                else:
                    deep_night = [tv_num]
                history_day_user.append(deep_night)

                if timeslot_1:
                    moring = list(set(timeslot_1))
                else:
                    moring = [tv_num]
                history_day_user.append(moring)

                if timeslot_2:
                    afternoon = list(set(timeslot_2))
                else:
                    afternoon = [tv_num]
                history_day_user.append(afternoon)

                if timeslot_3:
                    evening = list(set(timeslot_3))
                else:
                    evening = [tv_num]
                history_day_user.append(evening)

                history_day_user = tuple(history_day_user)  # 转换成元祖的形式
                history.append(history_day_user)

                month = int(date[0])
                day = int(date[1])
                timeslot_1 = []
                timeslot_2 = []
                timeslot_3 = []
                timeslot_4 = []

            # morining
            if hour >= 7 and hour <= 11:
                timeslot_1.append(tv_id)

            # afternoon
            if hour >= 12 and hour <= 18:
                timeslot_2.append(tv_id)

            # evening
            if hour >= 19 and hour <= 22:
                timeslot_3.append(tv_id)

            # dawn
            if (hour >= 0 and hour <= 6) or (hour == 23):
                timeslot_4.append(tv_id)
    f.close()

    # get consective period sample for one user
    new_history = []
    for i in range(len(history) - 1):
        tuple_pre = history[i]  # last period
        tuple_nex = history[i + 1]  # current period
        user_id_pre = tuple_pre[0]
        user_id_next = tuple_nex[0]

        if user_id_pre == user_id_next:
            next_mor_bas, next_aft_bas, next_eve_bas, next_deep_bas = tuple_nex[1:]
            user_tuple_pre = list(tuple_pre)
            user_tuple_pre.append(next_mor_bas)
            user_tuple_pre.append(next_aft_bas)
            user_tuple_pre.append(next_eve_bas)
            user_tuple_pre.append(next_deep_bas)
            new_history.append(tuple(user_tuple_pre))

    f = open('./data/tky/tky_for_our.pkl', 'wb')
    pickle.dump(new_history, f)
    f.close()


pre_tky()
