import scipy as sp
import numpy as np
import pandas as pd

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

def get_features(raw_fall_data, activity_dict, prev_activity_dict, sample_rate):
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
        Acc_max.append(np.array([max(abs(raw_fall_data.Acc[i][:,0])), max(abs(raw_fall_data.Acc[i][:,1])), max(abs(raw_fall_data.Acc[i][:,2]))]))
        Gyr_max.append(np.array([max(abs(raw_fall_data.Gyr[i][ :, 0])), max(abs(raw_fall_data.Gyr[i][ :, 1])), max(abs(raw_fall_data.Acc[i][ :, 2]))]))
        #Act.append(activity_dict[raw_fall_data.ActivityID[i]])
        Act.append(raw_fall_data.ActivityID[i])
        #prev_act.append(prev_activity_dict[raw_fall_data.ActivityID[i]])
        Acc_freq_max = np.array([max_freq(raw_fall_data.Acc[i][:,0],sample_rate),max_freq(raw_fall_data.Acc[i][:,1],sample_rate),max_freq(raw_fall_data.Acc[i][:,2],sample_rate)])
        Gyr_freq_max = np.array([max_freq(raw_fall_data.Gyr[i][:, 0],sample_rate), max_freq(raw_fall_data.Gyr[i][:, 1],sample_rate), max_freq(raw_fall_data.Gyr[i][:, 2],sample_rate)])
        Acc_freq.append(Acc_freq_max)
        Gyr_freq.append(Gyr_freq_max)
    Acc_max_w = []
    Gyr_max_w = []
    Acc_freq_w = []
    Gyr_freq_w = []
    Act_comp = []
    for i in range(len(raw_fall_data)):
        if dev[i] == 'Waist':
            Act_comp.append(Act[i])
            Acc_max_w.append(Acc_max[i])
            Gyr_max_w.append(Gyr_max[i])
            Acc_freq_w.append(Acc_freq[i])
            Gyr_freq_w.append(Gyr_freq[i])
    Acc_max_w = np.array(Acc_max_w)
    Gyr_max_w = np.array(Gyr_max_w)
    Acc_freq_w = np.array(Acc_freq_w)
    Gyr_freq_w = np.array(Gyr_freq_w)
    #Remember to add prev activity IDs
    feature_frame = pd.DataFrame(list(zip(Act_comp, Acc_max_w, Gyr_max_w, Acc_freq_w, Gyr_freq_w)), columns=['Activity_ID', 'W_Acc_max', 'W_Gyr_max', 'W_Acc_freq', 'W_Gyr_freq'])
    return feature_frame

def flatten_frame(data_set):
    flat_data_list = []
    for i in range(len(data_set)):
        entry_max = np.concatenate((np.asarray(data_set.W_Acc_max[i]),np.asarray(data_set.W_Gyr_max[i])))
        entry_freq = np.concatenate((data_set['W_Acc_freq'][i].flatten(),data_set['W_Gyr_freq'][i].flatten()))
        entry = np.concatenate((entry_max,entry_freq))
        flat_data_list.append(entry)
        print(flat_data_list[i].shape)
    flat_frame = pd.DataFrame(flat_data_list)
    return flat_frame

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

def process_raw(raw_data):
    flat_data_list = []
    Act_ID = []
    for i in range(len(raw_data)):
        if raw_data.Device[i] == 'Waist':
            entry = np.concatenate((np.asarray(raw_data.Acc[i]).flatten(),np.asarray(raw_data.Gyr[i]).flatten()))
            flat_data_list.append(entry)
            Act_ID.append(raw_data.ActivityID[i])
    flat_data_list =np.asarray(flat_data_list)
    flat_frame = pd.DataFrame(flat_data_list)
    return [flat_frame, Act_ID]