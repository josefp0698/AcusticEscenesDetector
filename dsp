# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:30:16 2021

@author: JOSE
"""
import numpy as np
from scipy import signal
### Module dsp ####


def Normalization_InputData(vrMean,vrStd,mrData):
    # Normalize data matriz with a median vector and a desviations vector. 
    # Inputs: 
    #      vrMean: median vector
    #      vrStd: desviations vector
    #      mrData: data oto normalize. 
    # Outputs: 
    #      mrData: normalization data matrix. 
    
    iRows,iColums =np.shape(mrData)
    counter = np.zeros(iRows)
    j = 0
    for i in counter:
     mrData[j,:] = np.divide((mrData[j,:]-vrMean),vrStd)
     j = j + 1
     
    return mrData

def enframe(vrAudio,iFrameLength):
    # Divides a signal into consecutive segments (frames) having length
    # iFrameLength and with a certain overlap between them (indicated by rOverlap).
    # Input parameters:
    #   vrSignal (required input): Vector signal to be split into frames.
    #   iFrameLength (required input): Frame length (number of samples).
    # Output parameters:
    #   mrSignal: Matrix with as many columns as signal frames and as many rows
    #       as samples per frame (each frame in one column).
    ###funcion enframe

    #Check inputs parameters
    vrAudio_type = type(vrAudio)
    iFrameLength_type = type(iFrameLength)
    if vrAudio_type != np.ndarray:
        raise TypeError("Error: audio type incorret, it must be a numpy array")
    if iFrameLength_type != int:
        raise TypeError("Error: The frame length must be int")
    dimensiones = np.shape(vrAudio)
 
    if len(dimensiones)>1:
        iRows, iColums = np.shape(vrAudio)
        if iRows == 2:
         vrAudio = vrAudio[1,:]
        if iColums == 2:
         vrAudio = vrAudio[:,1]
    
    
     
    iFrames = vrAudio.size 
    iSamples = int(iFrames/iFrameLength) 
    iSamplesCut=iSamples*iFrameLength 
    vrAudio_v = vrAudio[:iSamplesCut]
    vrSignal = np.reshape(vrAudio_v,(iSamples,iFrameLength))
    vrSignal = np.transpose(vrSignal)
    
    return vrSignal

def hpss(mrSignal_frame,iSamplingFreq = 48000,iMaxFrequency = 9000, sMask="none",inH = 5, inP = 5):
#   split the signal in two parts, mrH and mrP.
#   
# 
#  Input parameters:
#     mrSignal: Original signal, time domain.
#     iSamplingFreq: Sampling frequency
#     iMaxFrequency: Maximum spectrogram frequency  
#     sMask: Filter mask 
#     inH: nums of positions to calculate horizontal median filter
#     inP: nums of positions to calculate vertical median filter
#  Output parameters:
#    mrSiganlH: mrH component 
#    mrSiganlP: mrP component
# 
#  References:
#  [1] FitzGerald,DerrmrSpectogramPo "mrH/PERCUSSIVE SEPARATION USING MEDIAN
#  FILTERING" Audio Research Group, Dublin Institute of Technology. 
   
# Check input parameters. 
 if type (inH) not in (int, float):
    raise TypeError ("inH must be a scalar")
 if type (inP) not in (int, float):
    raise TypeError ("inP must be a scalar")
 if type (iMaxFrequency) not in (int, float):
    raise TypeError ("iMaxFrequency must be a scalar")
 if type (iSamplingFreq) not in (int, float):
    raise TypeError ("Frecuency must be a scalar")
 if type (mrSignal_frame) != np.ndarray:
    raise TypeError ("The input signal must be a vector")


 vrFrequency, vrTime, mrSpectrogram = signal.stft(mrSignal_frame, iSamplingFreq, 
                                                  nperseg=2205,window="hamming",nfft=2213,noverlap=0)
 
 mrSpectrogram  = np.abs(mrSpectrogram) 
 

 iDif = vrFrequency[1]-vrFrequency[0]
 iSample = round(iMaxFrequency/iDif)
 iSample = iSample - 1  
 mrSpectrogram = mrSpectrogram[:iSample,:]
 
 
 mrSpectogramPo = np.abs(mrSpectrogram)**2
 
 mrH = signal.medfilt(mrSpectogramPo, [1,inH])  
 mrP = signal.medfilt(mrSpectogramPo, [inP, 1])

   
 if sMask == "binary":
    maskH = np.int8(mrH >= mrP)
    maskP = np.int8(mrH < mrP)
    mrHPart = mrSpectrogram*maskH
    mrPPart = mrSpectrogram*maskP
    
 if sMask == "soft":
       iEps = 0.00001
       maskH = (mrH + iEps/2)/(mrH + mrP + iEps)
       maskP = (mrP + iEps/2)/(mrH + mrP + iEps)
       mrHPart = mrSpectrogram*maskH
       mrPPart = mrSpectrogram*maskP
 else: 
       mrHPart = mrP
       mrPPart = mrH
       
 return mrHPart,mrPPart


def Parameters_audio(mrAudio,iSamplingFreq):
    iFRAME_DURATION = 1.5
    inP = 5
    inH = 5

    
  
    iFrameLength = round(iSamplingFreq*iFRAME_DURATION)
    mrSignal_v = enframe(mrAudio,iFrameLength)
    iFrame=0
    
    
    for mrFrame in mrSignal_v[1,:]: 
       
       mrSignalH, mrSignalP = hpss(mrSignal_v[:,iFrame],iSamplingFreq,9000,"soft",inH,inP)
       iRows,iColums = np.shape(mrSignalH)
       mrSignalH = np.reshape(mrSignalH,[1,iRows*iColums])
       mrSignalP = np.reshape(mrSignalP,[1,iRows*iColums])
       if iFrame == 0:
           mrSignalH_m = mrSignalH
           mrSignalP_m = mrSignalP
       else:
           mrSignalH_m = np.vstack([mrSignalH_m, mrSignalH])
           mrSignalP_m = np.vstack([mrSignalP_m, mrSignalH])

           
       iFrame = iFrame + 1    
    mrSignal = np.concatenate((mrSignalH_m, mrSignalP_m), axis=1)

    
   
  
  
    
    return mrSignal
