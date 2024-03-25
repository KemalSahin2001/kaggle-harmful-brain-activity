import pandas as pd
def Create_non_overlapping_eegID_data(df, TARGETS):
    train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
    train.columns = ['spec_id','min']

    tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
        {'spectrogram_label_offset_seconds':'max'})
    train['max'] = tmp

    tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
    train['patient_id'] = tmp

    tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
    for t in TARGETS:
        train[t] = tmp[t].values
        
    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data

    tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train['target'] = tmp

    train = train.reset_index()
    print('Train non-overlapp eeg_id shape:', train.shape )

    return train, y_data
