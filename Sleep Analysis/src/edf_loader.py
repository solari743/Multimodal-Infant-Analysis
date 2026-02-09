import mne

def load_and_process_edf(edf_path, picks=None,l_freq=0.3, h_freq=35.0):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=True)

    if picks is not None:
        raw.picks(picks)


    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)