import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay

data_dir = 'WF'
score_format = 'WF'

features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'chroma', 'mfcc']
# features = ['stft', 'cqt', 'vqt', 'iirt', 'chroma', 'mfcc']
# features = ['stft', 'vqt', 'iirt', 'mfcc']

fig, ax = plt.subplots()

for feature in features:
    # for lang in ['cs', 'en']:
    #     gen_proto = open(
    #         f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/{data_dir}/{score_format}-{lang}-eval-{feature}-eval-genuine.txt')
    #     fake_proto = open(
    #         f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/{data_dir}/{score_format}-{lang}-eval-{feature}-eval-spoof.txt')
    #
    #     labels = []
    #     scores = []
    #
    #     for line in gen_proto:
    #         labels.append(1)
    #         scores.append(float(line))
    #
    #     for line in fake_proto:
    #         labels.append(0)
    #         scores.append(float(line))

    gen_proto = open(
        f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/{data_dir}/{score_format}-eval-{feature}-eval-genuine.txt')
    fake_proto = open(
        f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/{data_dir}/{score_format}-eval-{feature}-eval-spoof.txt')

    labels = []
    scores = []

    for line in gen_proto:
        labels.append(1)
        scores.append(float(line))

    for line in fake_proto:
        labels.append(0)
        scores.append(float(line))

    # DetCurveDisplay.from_predictions(labels, scores, ax=ax, name=feature.upper())
    RocCurveDisplay.from_predictions(labels, scores, ax=ax, name=feature.upper())

ax.plot([0, 1], [0, 1], '--', alpha=0.85, transform=ax.transAxes, linewidth=1)
ax.grid(linestyle="--")
plt.title("WaveFake")
plt.legend()
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(prop={'size': 12})
plt.show()
