
import scipy as sp
import numpy as np
import pandas as pd

#Extracts n highest freqiencies in the signal
#n determined by freq_select variable

def max_freq(data, sample_freq):
    freq_select = 6
    freq_bin = abs(sp.fft.fft(data))
    N = len(freq_bin)
    if (N % 2) == 0:
        mid_bin = int(N/2)
    else:
        mid_bin = (N+1)/2

    ind = np.argpartition(freq_bin[0:mid_bin], -freq_select)[-freq_select:]
    ind_sort = ind[np.argsort(freq_bin[ind])]

    max_freq_mag = freq_bin[ind_sort]

    max_freq = ind_sort*sample_freq/N

    return np.array([max_freq, max_freq_mag])
    #return max_freq


#Classifies activity into corresponding algorithm output
def act_class(ID):
    if ID >= 100:
        return True
    else:
        return False




#Gets features from data set
#These features based around mean and distribution features
#Still incomplete as it only looks at acceleration,but it seems much better - Robert
#Look at get_features for more comments on input
def get_features_alt(raw_fall_data, activity_dict, prev_activity_dict):

    Acc_std = []
    Acc_mean = []
    Acc_max = []
    Acc_min = []
    Acc_range = []
    Gyr_std = []
    Gyr_mean = []
    Gyr_max = []
    Gyr_min = []
    Gyr_range = []
    Act =[]

    win_time = 7
    sample_rate =238

    dev = raw_fall_data.Device
    trial_no = raw_fall_data.TrialNo
    sub_id = raw_fall_data.SubjectID

    for i in range(len(raw_fall_data)):

        center = np.argmax(np.sum(raw_fall_data.Acc[i]**2, axis= 1))
        ran = np.round(win_time / 2 * sample_rate)
        if center < ran:
            center = ran
        else:
            if (len(raw_fall_data.Acc[i][:, 2]) - center) < ran:
                center = len(raw_fall_data.Acc[i][:, 2]) - ran

        #Allows definition of start and stop to only look at part of signal, currently set to cover entire signal
        #start = 0
        #stop = len(raw_fall_data.Acc[i])

        start = int(center-ran)
        stop = int(center+ran)


        Acc_max_array = np.array([max(raw_fall_data.Acc[i][start:stop,0]), max(raw_fall_data.Acc[i][start:stop,1]), max(raw_fall_data.Acc[i][start:stop,2])])
        Acc_min_array = np.array(
            [min(raw_fall_data.Acc[i][start:stop, 0]), min(raw_fall_data.Acc[i][start:stop, 1]),
             min(raw_fall_data.Acc[i][start:stop, 2])])
        Acc_max.append(Acc_max_array)
        Acc_min.append(Acc_min_array)
        Acc_range.append(Acc_max_array - Acc_min_array)

        Gyr_max_array = np.array([max(raw_fall_data.Gyr[i][start:stop, 0]), max(raw_fall_data.Gyr[i][start:stop, 1]),
                                  max(raw_fall_data.Gyr[i][start:stop, 2])])
        Gyr_min_array = np.array(
            [min(raw_fall_data.Gyr[i][start:stop, 0]), min(raw_fall_data.Gyr[i][start:stop, 1]),
             min(raw_fall_data.Gyr[i][start:stop, 2])])
        Gyr_max.append(Gyr_max_array)
        Gyr_min.append(Gyr_min_array)
        Gyr_range.append(Gyr_max_array - Gyr_min_array)

        Acc_mean_array = np.array(
            [np.mean(raw_fall_data.Acc[i][start:stop, 0]), np.mean(raw_fall_data.Acc[i][start:stop, 1]),
             np.mean(raw_fall_data.Acc[i][start:stop, 2])])

        Acc_std_array = np.array(
            [np.std(raw_fall_data.Acc[i][start:stop, 0]), np.std(raw_fall_data.Acc[i][start:stop, 1]),
             np.std(raw_fall_data.Acc[i][start:stop, 2])])

        Gyr_mean_array = np.array(
            [np.mean(raw_fall_data.Gyr[i][start:stop, 0]), np.mean(raw_fall_data.Gyr[i][start:stop, 1]),
             np.mean(raw_fall_data.Gyr[i][start:stop, 2])])

        Gyr_std_array = np.array(
            [np.std(raw_fall_data.Gyr[i][start:stop, 0]), np.std(raw_fall_data.Gyr[i][start:stop, 1]),
             np.std(raw_fall_data.Gyr[i][start:stop, 2])])

        Acc_mean.append(Acc_mean_array)
        Acc_std.append(Acc_std_array)
        Gyr_mean.append(Gyr_mean_array)
        Gyr_std.append(Gyr_std_array)
        #Act.append(activity_dict[raw_fall_data.ActivityID[i]])
        Act.append(act_class(raw_fall_data.ActivityID[i]))
        #prev_act.append(prev_activity_dict[raw_fall_data.ActivityID[i]])

    Acc_max_w = []
    Gyr_max_w = []
    Acc_min_w = []
    Acc_mean_w = []
    Acc_std_w = []
    Acc_range_w = []
    Gyr_min_w = []
    Gyr_mean_w = []
    Gyr_std_w = []
    Gyr_range_w = []
    Act_comp = []
    prev_act_comp = []
    for i in range(len(raw_fall_data)):
        if dev[i] == 'Waist':
            Act_comp.append(Act[i])
            # prev_act_comp.append(prev_act[i])

            Acc_max_w.append(Acc_max[i])
            Acc_min_w.append(Acc_min[i])
            Acc_mean_w.append(Acc_mean[i])
            Acc_std_w.append(Acc_std[i])
            Acc_range_w.append(Acc_range[i])

            Gyr_max_w.append(Gyr_max[i])
            Gyr_min_w.append(Gyr_min[i])
            Gyr_mean_w.append(Gyr_mean[i])
            Gyr_std_w.append(Gyr_std[i])
            Gyr_range_w.append(Gyr_range[i])



    Acc_max_w = np.array(Acc_max_w)
    Acc_min_w = np.array(Acc_min_w)
    Acc_mean_w = np.array(Acc_mean_w)
    Acc_std_w = np.array(Acc_std_w)
    Acc_range_w = np.array(Acc_range_w)
    Gyr_max_w = np.array(Gyr_max_w)
    Gyr_min_w = np.array(Gyr_min_w)
    Gyr_mean_w = np.array(Gyr_mean_w)
    Gyr_std_w = np.array(Gyr_std_w)
    Gyr_range_w = np.array(Gyr_range_w)
    # Remember to add prev activity IDs
    # feature_frame = pd.DataFrame(list(zip(Act_comp,prev_act_comp,Acc_max_w,Gyr_max_w,Acc_freq_w,Gyr_freq_w)),
    #                            columns= ['Activity_ID','Prev_Act_ID','W_Acc_max','W_Gyr_max','W_Acc_freq','W_Gyr_freq'])

    feature_frame = pd.DataFrame(list(zip(Act_comp, Acc_max_w, Acc_min_w,Acc_mean_w,Acc_std_w,Acc_range_w, Gyr_max_w, Gyr_min_w,Gyr_mean_w,Gyr_std_w,Gyr_range_w)),
                                 columns=['Activity_ID', 'W_Acc_max', 'W_Acc_min','W_Acc_mean','W_Acc_std','W_Acc_range', 'W_Gyr_max', 'W_Gyr_min','W_Gyr_mean','W_Gyr_std','W_Gyr_range'])
    return feature_frame


#Gets features from data set
#These features based around frequency analysis
#I have note had much luck with this - Robert
#Although currently unused, I would like to use dictionaries to group similar activities together and identify previous activity as a feature
#For example, the classifier doesn't need to distinguish between different types of falls at al or at most just separate the ones with recovery from the others
def get_features(raw_fall_data, activity_dict, prev_activity_dict, sample_rate):


    win_time = 8
    Acc_max = []
    Gyr_max = []
    Act = []
    prev_act = []
    Acc_freq = []
    Gyr_freq = []
    dev = raw_fall_data.Device
    trial_no = raw_fall_data.TrialNo
    sub_id = raw_fall_data.SubjectID

    for i in range(len(raw_fall_data)):
        #Identifies window around y axis acceleration
        center = np.argmax(np.sum(raw_fall_data.Acc[i]**2, axis= 1))
        ran = np.round(win_time/2*sample_rate)
        if center<ran:
            center =ran;
        else:
            if (len(raw_fall_data.Acc[i][:, 2])-center) < ran:
                center = len(raw_fall_data.Acc[i][:, 2]) -ran

        # Allows definition of start and stop to only look at part of signal, currently set to look at a window around acceleration peak
        start = int(center-ran)
        stop = int(center+ran)
        #start = 0
        #stop = len(raw_fall_data.Acc)


        Acc_max.append(np.array([max(abs(raw_fall_data.Acc[i][start:stop,0])), max(abs(raw_fall_data.Acc[i][start:stop,1])), max(abs(raw_fall_data.Acc[i][start:stop,2]))]))
        Gyr_max.append(np.array([max(abs(raw_fall_data.Gyr[i][ start:stop, 0])), max(abs(raw_fall_data.Gyr[i][ start:stop, 1])), max(abs(raw_fall_data.Acc[i][ start:stop, 2]))]))
        #Act.append(activity_dict[raw_fall_data.ActivityID[i]])
        Act.append(act_class(raw_fall_data.ActivityID[i]))
        #prev_act.append(prev_activity_dict[raw_fall_data.ActivityID[i]])

        Acc_freq_max = np.array([max_freq(raw_fall_data.Acc[i][start:stop,0],sample_rate),max_freq(raw_fall_data.Acc[i][start:stop,1],sample_rate),max_freq(raw_fall_data.Acc[i][start:stop,2],sample_rate)])
        Gyr_freq_max = np.array([max_freq(raw_fall_data.Gyr[i][start:stop, 0],sample_rate), max_freq(raw_fall_data.Gyr[i][start:stop, 1],sample_rate), max_freq(raw_fall_data.Gyr[i][start:stop, 2],sample_rate)])

        Acc_freq.append(Acc_freq_max)
        Gyr_freq.append(Gyr_freq_max)

    Acc_max_w = []
    Gyr_max_w = []
    Acc_freq_w = []
    Gyr_freq_w = []
    Act_comp = []
    prev_act_comp = []
    for i in range(len(raw_fall_data)):
        if dev[i] == 'Neck':
            Act_comp.append(Act[i])
            #prev_act_comp.append(prev_act[i])

            Acc_max_w.append(Acc_max[i])
            Gyr_max_w.append(Gyr_max[i])
            Acc_freq_w.append(Acc_freq[i])
            Gyr_freq_w.append(Gyr_freq[i])


    Acc_max_w = np.array(Acc_max_w)
    Gyr_max_w = np.array(Gyr_max_w)
    Acc_freq_w = np.array(Acc_freq_w)
    Gyr_freq_w = np.array(Gyr_freq_w)
    #Remember to add prev activity IDs
    #feature_frame = pd.DataFrame(list(zip(Act_comp,prev_act_comp,Acc_max_w,Gyr_max_w,Acc_freq_w,Gyr_freq_w)),
    #                            columns= ['Activity_ID','Prev_Act_ID','W_Acc_max','W_Gyr_max','W_Acc_freq','W_Gyr_freq'])

    feature_frame = pd.DataFrame(list(zip(Act_comp, Acc_max_w, Gyr_max_w, Acc_freq_w, Gyr_freq_w)),
                                 columns=['Activity_ID', 'W_Acc_max', 'W_Gyr_max', 'W_Acc_freq',
                                          'W_Gyr_freq'])
    return feature_frame

#Flattens frame of frequency feature extraction so it can be used by classifier
def flatten_frame(data_set):

    flat_data_list = []

    for i in range(len(data_set)):
        entry_Acc = np.concatenate((np.asarray(data_set.W_Acc_max[i]),data_set['W_Acc_freq'][i].flatten()))
        entry_Gyr = np.concatenate((np.asarray(data_set.W_Gyr_max[i]),data_set['W_Gyr_freq'][i].flatten()))
        entry = np.concatenate((entry_Acc,entry_Gyr))
        flat_data_list.append(entry)
        print(flat_data_list[i].shape)

    flat_frame = pd.DataFrame(flat_data_list)
    return flat_frame
#Alternate version of flatten_frame for alternative feature extraction
def flatten_frame_alt(data_set):

    flat_data_list = []

    for i in range(len(data_set)):
        entry_max = np.concatenate((np.asarray(data_set.W_Acc_max[i]),np.asarray(data_set.W_Acc_min[i])))
        entry_var = np.concatenate((np.asarray(data_set.W_Acc_mean[i]),np.asarray(data_set.W_Acc_std[i])))
        entry_acc = np.concatenate((entry_max,entry_var,data_set.W_Acc_range[i]))

        entry_max = np.concatenate((np.asarray(data_set.W_Gyr_max[i]), np.asarray(data_set.W_Gyr_min[i])))
        entry_var = np.concatenate((np.asarray(data_set.W_Gyr_mean[i]), np.asarray(data_set.W_Gyr_std[i])))
        entry_gyr = np.concatenate((entry_max, entry_var, data_set.W_Gyr_range[i]))

        entry = np.concatenate((entry_acc,entry_gyr))
        entry2 = np.concatenate((entry_acc, entry_max))
        flat_data_list.append(entry_acc)
        print(flat_data_list[i].shape)

    flat_frame = pd.DataFrame(flat_data_list)
    return flat_frame

#Testng more general flatten function (currently unused)
def flat_entry(sub_entry):
    tup_len = len(sub_entry.shape)

    entry_len = len(sub_entry)
    out = []
    if tup_len == 1:
        for i in range(entry_len):
            out.append(sub_entry[i])


    if tup_len ==2:
        for i in range(sub_entry.shape[1]):
            for j in range(entry_len):
                out.append(sub_entry[i,j])

    if tup_len ==3:
        for i in range(sub_entry.shape[2]):
            for j in range(sub_entry.shape[1]):
                for k in range(entry_len):
                    out.append(sub_entry[i,j,k])

    return np.array(out)

#Just extracts the raw sensor data to run through classifier (Not effective)
def process_raw(raw_data):

    flat_data_list = []
    Act_ID = []

    for i in range(len(raw_data)):
        if raw_data.Device[i] == 'Waist':
            entry = np.concatenate((np.asarray(raw_data.Acc[i]).flatten(),np.asarray(raw_data.Gyr[i]).flatten()))
            flat_data_list.append(entry)
            Act_ID.append(raw_data.ActivityID[i])
        #print(flat_data_list[i].shape)

    flat_data_list =np.asarray(flat_data_list)

    flat_frame = pd.DataFrame(flat_data_list)
    return [flat_frame, Act_ID]
