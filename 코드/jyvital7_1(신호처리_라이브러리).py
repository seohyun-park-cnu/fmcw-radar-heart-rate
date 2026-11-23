'''jyvital'''
# 2023-05-30 : Made by Jaeyoung Choi
# 2023-06-20 : Add(BR), ErrorFix(stft_bpm : measuretime -> 60)
# 2023-08-16 : Kalman Filter Update
# 2023-09-04 : version2 is released
# 2024-04-01 : version3 is released (Small Movement Cancelation, DC Compensation)
# 2024-04-15 : version4 is released (DC Compensation -> MTI)
# 2024-04-17 : Peak Search
# 2024-06-26 : DACM Update
# 2024-07-05 : CancelMotion_SF is deleted
# 2024-07-05 : CancelMotion_FR -> Moving Average
# 2024-07-08 : Block Avg, Weighted Block Avg, SMA, WMA
# 2024-07-15 : Version7 is released (STFT + Zero-padding, Block Average is simplified)

import numpy as np
from scipy.signal import butter, filtfilt
from scipy import stats

def phase_unwrapping(target):  # Not Recommended Method
    phase = np.angle(target)
    for i in range(1,len(phase)):
        if phase[i] - phase[i-1] > np.pi:
            phase[i] = phase[i] - np.pi*2
        elif phase[i] - phase[i-1] < -np.pi:
            phase[i] = phase[i] + np.pi*2
        else:
            pass
    return phase

def dacm_phase_unwrapping(target):
    delta_phase = np.zeros(len(target))
    phase_dacm = np.zeros(len(target))
    I = target.real
    Q = target.imag
    for i in range(1,len(delta_phase)):
        delta_phase[i] = (I[i]*(Q[i]-Q[i-1])-Q[i]*(I[i]-I[i-1])) / ((I[i])**2+(Q[i])**2)
        phase_dacm[i] = phase_dacm[i-1]+delta_phase[i]
    return phase_dacm

def phase_differential(phase):
    phase_diff = np.zeros(len(phase))
    for i in range(1,len(phase)):
        phase_diff[i] = phase[i] - phase[i-1]
    return phase_diff

def kalman_filter(data,R=0.1**2):
    sz = (len(data),)
    Q = 1e-7   # process noise: changed 1e-5 -> 1e-7 (backend operating point, R=0.10)
    xhat = np.zeros(sz)
    P = np.zeros(sz)
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)
    '''Default : R = 0.1**2 '''
    xhat[0] = data[0]   # 0.0
    P[0] = 1.0
    for i in range(1,len(data)):
        xhatminus[i] = xhat[i-1]
        Pminus[i] = P[i-1] + Q
        K[i] = Pminus[i] / (Pminus[i] + R)
        xhat[i] = xhatminus[i] + K[i] * (data[i] - xhatminus[i])
        P[i] = (1 - K[i]) * Pminus[i]
    return xhat

def phase_stft(phase,W,L,numFrame):
    k = int(np.floor( (len(phase)-L) / (W-L) )) # time sample
    phase_stft = np.zeros((numFrame,k), dtype=complex)
    for i in range(k):
        phase_stft[:,i] = np.fft.fft(phase[ (W-L)*i : W+(W-L)*i ], n=numFrame)
    return phase_stft

def stft_Hz(numFrame,fps):
    stft_Hz = fps/numFrame * np.arange(numFrame)
    return stft_Hz

def stft_bpm(numFrame,measuretime,fps):
    stft_bpm = fps*measuretime/numFrame * np.arange(numFrame)
    return stft_bpm

def stft_time(numFrame,W,L,fps):
    k = int(np.floor( numFrame-L) / (W-L) )
    stft_time = np.arange(k)/fps + W/fps
    return stft_time

def bandpass_filter(data, lowcut, highcut, fps, order=4):
    '''
    normal : 0.8 - 2.0
    doll : 0.5 - 1.3
    jy : 0.8 - 2.1
    '''
    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def lowpass_filter(data, cutoff, fps, order=4):
    nyquist = 0.5 * fps
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def MTI(data):                      # Frame, Direction, Sample
    numFrame = len(data[:,0,0])
    numDirection = len(data[0,:,0])
    numSample = len(data[0,0,:])
    clutter = np.zeros((numFrame,numDirection,numSample),dtype=complex)
    mti = np.zeros((numFrame,numDirection,numSample),dtype=complex)
    alpha = 0.01    # 0.01
    for i in range(numDirection):
        for j in range(1,numFrame):
            clutter[j,i,:] = alpha * data[j,i,:] + (1-alpha) * clutter[j-1,i,:]
            mti[j,i,:] = data[j,i,:] - clutter[j,i,:]
    return mti

def PeakSearch(data,mindirec=0,maxdirec=15,minrange=0,maxrange=31):
    # data : frames, directions, rangebins
    datalength = data.shape[0]
    peak = np.zeros((datalength,2),dtype=int)
    for i in range(datalength):
        direcpeak = np.where(np.abs(data[i,mindirec:maxdirec+1,minrange:maxrange+1])==np.max(np.abs(data[i,mindirec:maxdirec+1,minrange:maxrange+1])))[0][0] + mindirec
        rangepeak = np.where(np.abs(data[i,mindirec:maxdirec+1,minrange:maxrange+1])==np.max(np.abs(data[i,mindirec:maxdirec+1,minrange:maxrange+1])))[1][0] + minrange
        peak[i,:] = np.array([direcpeak,rangepeak])
    # peakfix = [stats.mode(peak)[0][0],stats.mode(peak)[0][1]]
    return peak

def heartbeat_estimation(hr_stft,STFT_bpm,stft_frame_length,kalmanR=0.1**2):
    estim = np.zeros(stft_frame_length, dtype=float)
    for i in range(stft_frame_length):
        estim[i] = (STFT_bpm[np.argmax(np.abs(hr_stft[np.where((STFT_bpm<=180)&(STFT_bpm>=0)),i]))])
    estim_kalman = kalman_filter(estim,R=kalmanR)
    return estim_kalman
    # return estim

def heartbeat_estimation_with_harmonic_check(hr_stft, STFT_bpm, stft_frame_length,
                                             resp_rate_hz, kalmanR=0.1**2,
                                             harmonic_tol_bpm=2.0):
    """
    호흡 조파(Harmonic) 검증을 포함한 심박수 추정.

    STFT 각 프레임에서 피크 후보를 내림차순으로 검토하여, 호흡 주파수의
    정수배(n × f_resp)와 겹치는 후보를 조파로 판단하고 차순위 후보를 선택한다.
    모든 후보가 조파와 겹칠 경우 1순위(argmax)로 fallback한다.

    Args:
        hr_stft: STFT 결과 배열 (numFrame × stft_frame_length).
        STFT_bpm: BPM 축 배열.
        stft_frame_length: STFT 시간 프레임 수.
        resp_rate_hz: 추정된 호흡 주파수 (Hz).
        kalmanR: 칼만 필터 측정 노이즈.
        harmonic_tol_bpm: 조파 겹침 판단 허용 오차 (BPM).

    Returns:
        np.array: 칼만 필터링된 BPM 추정값 시계열.
    """
    resp_bpm = resp_rate_hz * 60.0
    # 1~5차 조파 BPM 목록 생성
    harmonic_bpms = [resp_bpm * n for n in range(1, 6)]

    hr_band_mask = (STFT_bpm >= 0) & (STFT_bpm <= 180)
    bpm_axis = STFT_bpm[hr_band_mask]

    estim = np.zeros(stft_frame_length, dtype=float)

    for i in range(stft_frame_length):
        spectrum_slice = np.abs(hr_stft[hr_band_mask, i])
        sorted_idx = np.argsort(spectrum_slice)[::-1]

        selected_bpm = None
        for idx in sorted_idx[:5]:  # 상위 5개 후보 순서대로 검토
            candidate = bpm_axis[idx]
            is_harmonic = any(
                abs(candidate - h) < harmonic_tol_bpm
                for h in harmonic_bpms
            )
            if not is_harmonic:
                selected_bpm = candidate
                break

        # 모든 후보가 조파와 겹치면 1순위 사용 (fallback)
        estim[i] = selected_bpm if selected_bpm is not None else bpm_axis[sorted_idx[0]]

    return kalman_filter(estim, R=kalmanR)

def heartbeat_estimation_adaptive_window(
    hr_stft, STFT_bpm, stft_frame_length, kalmanR=0.10,
    warmup_frames=200, prior_low=50.0, prior_high=120.0,
    initial_half_width=40.0, min_half_width=30.0, max_half_width=40.0,
    stability_window=30, stable_std_threshold=5.0, unstable_std_threshold=12.0,
    abs_floor=40.0, abs_ceil=130.0, narrow_rate=0.2, widen_rate=2.0,
    proximity_sigma=0.0,
):
    """
    Adaptive Window STFT Peak Picking with physiological prior + Kalman-guided tracking.

    Phase 1 (warmup, frames 0 ~ warmup_frames-1):
      Fixed search range [prior_low, prior_high] BPM.
      Blocks DC peaks (0-10 BPM) and respiration harmonics (12-24 BPM).

    Phase 2 (tracking, frames warmup_frames ~ end):
      center = online Kalman prediction (xhat from previous frame).
      search range = [max(abs_floor, center-hw), min(abs_ceil, center+hw)].
      Half-width adapts: narrow when stable (std < 5), widen when unstable (std > 12).

    proximity_sigma: if > 0, apply Gaussian proximity weighting to prefer peaks
      closer to the Kalman center. Effective during tracking phase only.
      sigma=10 gives strong preference within ±10 BPM of center.

    Returns: kalman_filter(raw_estim, R=kalmanR)
    """
    raw_estim = np.zeros(stft_frame_length, dtype=float)

    # Online Kalman state for center tracking (separate from output Kalman)
    # Q_online=0.01 yields K≈0.24 (responsive tracking, ~1s time constant)
    Q_online = 0.01
    xhat = 0.0
    P = 1.0

    hw = initial_half_width
    effective_warmup = min(warmup_frames, stft_frame_length)
    use_proximity = proximity_sigma > 0.0

    for i in range(stft_frame_length):
        # --- Determine search range ---
        if i < effective_warmup:
            lo, hi = prior_low, prior_high
        else:
            if i == effective_warmup:
                # Transition: initialize center from warmup median
                center = float(np.median(raw_estim[:effective_warmup]))
                xhat = center
                P = 1.0
                hw = initial_half_width

            center = xhat
            lo = max(abs_floor, center - hw)
            hi = min(abs_ceil, center + hw)

            # Adapt half-width based on recent estimation stability
            if i >= effective_warmup + stability_window:
                recent_std = float(np.std(raw_estim[i - stability_window:i]))
                if recent_std < stable_std_threshold:
                    hw = max(min_half_width, hw - narrow_rate)
                elif recent_std > unstable_std_threshold:
                    hw = min(max_half_width, hw + widen_rate)

        # --- Peak picking within [lo, hi] ---
        band_mask = (STFT_bpm >= lo) & (STFT_bpm <= hi)
        if np.any(band_mask):
            spectrum = np.abs(hr_stft[band_mask, i])
            bpm_candidates = STFT_bpm[band_mask]
            if use_proximity and i >= effective_warmup:
                # Gaussian proximity weighting: prefer peaks near Kalman center
                proximity = np.exp(-0.5 * ((bpm_candidates - xhat) / proximity_sigma) ** 2)
                raw_estim[i] = bpm_candidates[np.argmax(spectrum * proximity)]
            else:
                raw_estim[i] = bpm_candidates[np.argmax(spectrum)]
        else:
            # Fallback: online Kalman value or prior midpoint
            if i >= effective_warmup:
                raw_estim[i] = xhat
            else:
                raw_estim[i] = (prior_low + prior_high) / 2.0

        # --- Update online Kalman ---
        if i == 0:
            xhat = raw_estim[0]
            P = 1.0
        else:
            xhatminus = xhat
            Pminus = P + Q_online
            K = Pminus / (Pminus + kalmanR)
            xhat = xhatminus + K * (raw_estim[i] - xhatminus)
            P = (1 - K) * Pminus

    return kalman_filter(raw_estim, R=kalmanR)

def stft_normalization(stft_data):
    stft_norm = np.zeros((len(stft_data[:,0]),len(stft_data[0,:])))
    for i in range(len(stft_data[0,:])):
        stft_norm[:,i] = np.abs(stft_data[:,i]) / np.max(np.abs(stft_data[:,i]))
    return stft_norm
    