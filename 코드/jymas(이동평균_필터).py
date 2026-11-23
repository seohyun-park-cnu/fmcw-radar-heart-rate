'''jymas'''
# 2024-08-30 : Made by Jaeyoung Choi


import numpy as np

def sma(phase,size):
    '''Simple Moving Average'''
    length = len(phase)
    result = np.zeros(length)
    result[:size] = phase[:size]
    for i in range(size,length-size):
        result[i] = np.mean(phase[i-size:i+size+1])
    result[length-size:] = phase[length-size:]
    return result

def wma(phase,size):
    length = len(phase)
    result = np.zeros(length)
    result[:size] = phase[:size]
    wL = np.arange(1,size+1,1)
    wC = np.array([size+1])
    wR = np.arange(size,0,-1)
    weight = np.concatenate((wL,wC,wR))
    for i in range(size,length-size):
        result[i] = np.sum(weight*phase[i-size:i+size+1]) / np.sum(weight)
    result[length-size:] = phase[length-size:]
    return result

# def ema(phase,size=10,alpha=2/(11+1)):
#     length = len(phase)
#     emaForward = np.zeros(length)
#     emaBackward = np.zeros(length)
#     result = np.zeros(length)
#     emaForward[:size] = phase[:size]                # 0~10
#     emaBackward[:size] = np.flip(phase[:size])      # 10~0
#     result[:size] = phase[:size]                    # 0~10
#     for i in range(size,length-size):
#         phaseForward = phase[i-size:i+1]                # 0~10, 1~11, ...
#         phaseBackward = np.flip(phase[i:i+size+1])      # 10~0, 11~1, ...
#         emaForward[i] = alpha*phaseForward[size] + (1-alpha)*emaForward[i-1]
#         emaBackward[i] = alpha*phaseBackward[size] + (1-alpha)*emaBackward[i-1]
#         result[i] = (emaForward[i]+emaBackward[i])/2
#     result[length-size:] = phase[length-size:]
#     return result

# def ema(phase,size=10):
#     alpha = 2/(size+1)
#     length = len(phase)
#     forward = np.zeros(length)
#     backward = np.zeros(length)
#     result = np.zeros(length)
#     forward[0] = phase[0]
#     backward[0] = phase[2]
#     for i in range(1,length-1):
#         forward[i] = alpha*phase[i] + (1-alpha)*forward[i-1]
#         backward[i] = alpha*phase[i] + (1-alpha)*backward[i-1]
#         result[i] = (forward[i]+backward[i])/2
#     result[length-1] = phase[length-1]
#     return result

def ema(phase,size):
    length = len(phase)
    alpha = 2 / (size + 1)
    forward_ema = np.zeros(length)
    forward_ema[0] = phase[0]
    for i in range(1,length):
        forward_ema[i] = alpha * phase[i] + (1 - alpha) * forward_ema[i-1]
    backward_ema = np.zeros(length)
    backward_ema[-1] = phase[-1]
    for i in range(length-2, -1, -1):
        backward_ema[i] = alpha * phase[i] + (1 - alpha) * backward_ema[i+1]
    result = (forward_ema + backward_ema) / 2
    return result
