import hmm
import csv
import numpy as np
import xml.etree.ElementTree as et
from sklearn.metrics import accuracy_score

np.random.seed(47)

filenames = [20081009, 20081010, 20081011, 20081012, 20081013, 20081017, 20081018,
         20081019, 20081020, 20081021, 20081023, 20081023, 20081024, 20081025]

diff = []
state_seq = []

# Obtain a vector (diff) of euclidean distances between successive positions
# and round them to 3 significant digits.
# Obtain a vector (state_seq) of labels. These vectors are the same length.
for filename in filenames:
    with open('gpsdata/' + str(filename) + '.txt', 'r') as f:
        rows = [row.split() for row in f]

    root = et.parse('gpsdata/' + str(filename) + '.gpx').getroot()
    for i,trk in enumerate(root):
        name = trk[0].text[10:]
        assert name == rows[i][0]
        pos = None
        for trkpt in trk[1]:
            npos = np.array([float(trkpt.attrib['lat']),
                             float(trkpt.attrib['lon'])])
            if pos is not None:
                diff.append("{0:.3f}".format(np.linalg.norm(npos - pos)))
                state_seq.append(rows[i][1])
            pos = npos

# Compute a vector (observations) of every possible differences rounded to 3
# significant digits and compute a vector (observ_seq) of observed outputs as
# the index of the observed difference in array observations.
observations = [o for o in set(diff)]
observ_seq = [observations.index(d) for d in diff]

# Train HMM using Baum-Welch algorithm.
n_obs = len(observations)
model = hmm.HMM(3, n_obs)
model.train([observ_seq])

# Compute the most probable sequence of hidden states using Viterbi algorithm.
pred_state_seq = model.viterbi(observ_seq)

# Compute the accuracy of the prediction.
# TODO: map hidden states
# print("Accuracy: {}".format(accuracy_score(state_seq, pred_state_seq)))
