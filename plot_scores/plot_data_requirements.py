import matplotlib.pyplot as plt
import numpy as np

features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'mfcc', 'chroma']
sample_sizes = np.array([31.0, 62.0, 10.2, 10.2, 11.7, 2.5, 1.5])
ram_usage = np.array([82, 200, 39, 42, 55, 14, 11])
extraction_time = np.array([0.006126217865724557, 0.0008459329098905981, 0.02589292418256518, 0.023399459328621055,
                            0.16722976258690275, 0.005534416888507646, 0.0038180231457472015])

normalizedSizes = sample_sizes/np.max(sample_sizes)
normalizedRam = ram_usage/np.max(ram_usage)
normalizedTime = extraction_time/np.max(extraction_time)

x_axis = np.arange(len(features))

plt.bar(x_axis - 0.3, normalizedSizes, 0.3, label='Storage')
plt.bar(x_axis, normalizedRam, 0.3, label='RAM')
plt.bar(x_axis + 0.3, normalizedTime, 0.3, label='Extraction time')
plt.xticks(x_axis, features)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(prop={'size': 12})
plt.xlabel('Spectrogram (detector)', fontsize=12)
plt.ylabel('Max-normalized value', fontsize=12)
plt.tight_layout()
plt.show()

# mel: total - 8.656345844268799: per sample - 0.006126217865724557
# stft: total - 1.195303201675415: per sample - 0.0008459329098905981
# cqt: total - 36.5867018699646: per sample - 0.02589292418256518
# vqt: total - 33.06343603134155: per sample - 0.023399459328621055
# iirt: total - 236.29565453529358: per sample - 0.16722976258690275
# mfcc: total - 7.820131063461304: per sample - 0.005534416888507646
# chroma: total - 5.394866704940796: per sample - 0.0038180231457472015