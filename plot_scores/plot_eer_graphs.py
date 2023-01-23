import matplotlib.pyplot as plt

features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'mfcc', 'chroma']

# EER rates retrieved from pyeeer report (.csv) from validation of each dataset
eer_asv2019 = [0.2559, 0.2036, 0.1967, 0.2073, 0.2656, 0.1571, 0.2729]
eer_f2s = [0.0559, 0.0106, 0.0346, 0.0417, 0.0728, 0.0070, 0.2661]
eer_frec = [0.0160, 0.0045, 0.0035, 0.0058, 0.0068, 0.0019, 0.1055]
eer_wf = [0.0055, 0.0004, 0.0089, 0.0018, 0.1665, 0.0039, 0.3225]
eer_avg = [0.1264, 0.1003, 0.0971, 0.0994, 0.2044, 0.0761, 0.2804]

# plot line
plt.plot(features, eer_asv2019, marker='x', label="AS19")
plt.plot(features, eer_f2s, marker='x', label="F2S")
plt.plot(features, eer_frec, marker='x', label="FREC")
plt.plot(features, eer_wf, marker='x', label="WF")
plt.plot(features, eer_avg, 'b--', marker='x', label="Total")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(prop={'size': 12})
plt.ylabel('EER', fontsize=12)
plt.xlabel('Spectrogram (detector)', fontsize=12)
plt.tight_layout()
plt.show()
