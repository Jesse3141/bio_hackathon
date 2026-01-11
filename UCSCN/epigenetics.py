# epigenetics.py
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com


from json import load
import pandas as pd
import numpy as np
import itertools as it
import seaborn as sns
from matplotlib import pyplot as plt
import re
import sys
import torch
import time

from functools import reduce

from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

from yahmm_loader import load_yahmm_model
# from PyPore.hmm import *
# from PyPore.DataTypes import *


def EpigeneticsModel(distributions, name, low=0, high=90):
    """
    Create the HMM using circuit board methodologies.
    """

    def match_model(distribution, name):
        """
        Build a small match model, allowing for oversegmentation where the
        distribution representing number of segments is a mixture of two
        exponentials.
        """
        distributions = [distribution, distribution]

        transitions = torch.tensor([[0.10, 0.00], [0.00, 0.80]])

        starts = torch.tensor([0.95, 0.05])
        ends = torch.tensor([0.90, 0.20])

        model = DenseHMM(
            distributions=distributions,
            starts=starts,
            ends=ends,
            edges=transitions,
            verbose=True,
            max_iter=10,
        )

        return model


def build_profile():
    """
    Build a profile HMM based on a file of hand curated data, with forks in the
    appropriate places.

    This is a cartoon of the HMM:

       /---mC---\        /---mC---\   /-CAT-\
    ---|----C---|-------|----C---|---|--T--|-----------
       \--hmC---/        \--hmC---/   \--X--/
    
    """

    data = pd.read_excel("data.xlsx", "Sheet1")
    hmms, dists = [], {}

    for name, frame in data.groupby("Label"):
        means, stds = frame.mean(axis=0), frame.std(axis=0)
        dists[name] = [Normal(m, s) for m, s in zip(means, stds) if not np.isnan(m)]

    labels = ["T", "CAT", "X"]
    cytosines = ["C", "mC", "hmC"]

    profile = []
    profile.extend(dists["CAT"][::-1])

    for i in range(9):
        profile.append(
            {
                "C": dists["C"][8 - i],
                "mC": dists["mC"][8 - i],
                "hmC": dists["hmC"][8 - i],
            }
        )

    profile.extend(dists["CAT"][::-1])
    profile.extend(dists["CAT"][::-1])
    profile.extend(dists["CAT"])
    profile.extend(dists["CAT"])

    for i in range(9):
        profile.append(
            {"C": dists["C"][i], "mC": dists["mC"][i], "hmC": dists["hmC"][i]}
        )

    profile.extend(dists["CAT"])

    for i in range(6):
        profile.append(
            {"T": dists["T"][i], "X": dists["X"][i], "CAT": dists["CAT"][i % 3]}
        )

    profile.extend(dists["CAT"])
    profile.extend(dists["CAT"])
    profile.extend(dists["CAT"])
    profile.extend(dists["CAT"])

    return profile


def analyze_events(events, hmm):
    """
    Take in a list of events and create a dataframe of the data.
    """

    data = {}
    data["Filter Score"] = []
    data["C"] = []
    data["mC"] = []
    data["hmC"] = []
    data["X"] = []
    data["T"] = []
    data["CAT"] = []
    data["Soft Call"] = []

    tags = ("C", "mC", "hmC", "X", "T", "CAT")
    indices = {state.name: i for i, state in enumerate(hmm.states)}

    for idx, event in enumerate(events):
        # Hold data for the single event in 'd'. The fields will hold a single
        # piece of information.
        d = {key: None for key in data.keys()}

        # Run forward-backward on that event to get the expected transitions
        # matrix
        # print(event)
        trans, ems = hmm.forward_backward(event)

        # Get the expected number of transitions from all states to the first
        # match state of each fork, and the the expected number of transitions
        # from the last match state to the closing of the fork.
        # The measurement we use for this is the minimum of the expected
        # transitions into the fork and the expected transitions out of the
        # fork, because we care about the expectation of going through the
        # entire event.
        for tag in "C", "mC", "hmC":
            names = ["M-{}:{}-end".format(tag, i) for i in range(25, 34)]
            d[tag] = min([trans[indices[name]].sum() for name in names])

        # Perform the same calculation, except on the label fork now instead
        # of the cytosine variant fork.
        for tag in "X", "T", "CAT":
            names = ["M-{}:{}-end".format(tag, i) for i in range(37, 43)]
            d[tag] = min([trans[indices[name]].sum() for name in names])

        # Calculate the score, which will be the sum of all expected
        # transitions to each of the three forks, and the expected transitions
        # into each of the three labels. This gives us a score representing
        # how likely it is that this event went through the fork and through
        # the label.
        d["Filter Score"] = sum(d[tag] for tag in tags[:3]) * sum(
            d[tag] for tag in tags[3:]
        )

        # Calculate the dot product score between the posterior transition
        # probability of transitions for cytosine variants, and the
        # corresponding labels
        score = d["C"] * d["T"] + d["mC"] * d["CAT"] + d["hmC"] * d["X"]
        d["Soft Call"] = score / d["Filter Score"] if d["Filter Score"] != 0 else 0

        for key, value in d.items():
            data[key].append(value)

    return pd.DataFrame(data)


def insert_delete_plot(model, events):
    """
    Calculate the probability of an insert or delete at each position in the
    HMM.
    """

    # Get the length of the profile from the HMM name
    n = int(model.name.split("-")[-1])

    # Get a mapping from states to indices in the state list
    indices = {state.name: i for i, state in enumerate(model.states)}
    delete_names = [state.name for state in model.states if "D" in state.name]
    insert_names = [state.name for state in model.states if "I" in state.name]
    backslip_names = [
        state.name
        for state in model.states
        if "b" in state.name and state.name[-2:] == "e7"
    ]
    underseg_names = [
        state.name
        for state in model.states
        if "U" in state.name and "s" not in state.name and "e" not in state.name
    ]
    array_names = [delete_names, insert_names, backslip_names, underseg_names]

    # Create a list of expected transitions to deletes and inserts respectively
    deletes = np.zeros(n + 1)
    inserts = np.zeros(n + 1)
    backslips = np.zeros(n + 1)
    undersegmentation = np.zeros(n + 1)
    arrays = [deletes, inserts, backslips, undersegmentation]

    # Go through each event, calculating the number of transitions
    for event in events:
        # Run forward-backward to get the expected transition counts
        trans, ems = model.forward_backward(event)

        # For each delete state, add the number of expected transitions
        # for this event to the total running count
        for array, names in zip(arrays, array_names):
            for name in names:
                if array is backslips:
                    position = int(name.split(":")[-1].split("e")[0])
                else:
                    position = int(name.split(":")[-1])
                index = indices[name]

                array[position] += trans[:, index].sum()

    for array in arrays:
        array /= float(len(events))

    plt.subplot(411)
    plt.bar(np.arange(n + 1) + 0.2, inserts, 0.8, color="c", alpha=0.66)

    plt.subplot(412)
    plt.bar(np.arange(n + 1) + 0.2, deletes, 0.8, color="m", alpha=0.66)

    plt.subplot(413)
    plt.bar(np.arange(n + 1) + 0.2, backslips, 0.8, color="#FF6600", alpha=0.66)

    plt.subplot(414)
    plt.bar(np.arange(n + 1) + 0.2, undersegmentation, 0.8, color="g", alpha=0.66)

    plt.show()


def train(model: DenseHMM, events, threshold=0.10):
    """
    This is a full step of training of the model. This involves a cross-training
    step to determine which parameter should be used to filter out events from
    the full training step, and then 10 iterations of Baum-Welch training. This
    assumes that all events passed in are training events.
    """

    tic = time.time()
    data = analyze_events(events, model)
    print("Scoring Events Took {}s".format(time.time() - tic))

    # Get the events using the best threshold
    events = [
        event for score, event in zip(data["Filter Score"], events) if score > threshold
    ]
    print("{} events after filtering".format(len(events)))

    # Train the HMM using those events
    tic = time.time()
    X = torch.tensor(events)
    # max_iter was set in the model initialisation
    model.fit(X)
    print("Training on {} events took {}".format(len(events), time.time() - tic))

    return model


def test(model, events):
    """
    Test the model on the events, returning a list of accuracies should the top
    i events be used.
    """

    # Analyze the data, and sort it by filter score
    data = analyze_events(events, model).sort("Filter Score")
    n = len(data)

    # Attach the list of accuracies using the top i events to the frame
    data["MCSC"] = [1.0 * sum(data["Soft Call"][i:]) / (n - i) for i in range(n)]

    # Return the frame
    return data


def n_fold_cross_validation(events, n=5):
    """
    Perform n-fold cross-validation, wherein the data is split into n equally
    sized sections, and the model is trained on all but one of them, and
    classifies the last one. This is repeated until each section has been
    classified. For each iteration, the model is pulled fresh from a text
    document to ensure that it is not modified by any round of training.
    """

    # Divide the data into n equally sized folds
    folds = [events[i::n] for i in range(n)]

    # Go through each one, defining which one is for testing and which ones
    # are for training.
    for i in range(n):
        training = reduce(lambda x, y: x + y, folds[:i] + folds[i + 1 :], [])
        testing = folds[i]

        model, _ = load_yahmm_model("untrained_hmm.txt", False)
        model = train(model, training, threshold=0.1)

        if i == 0:
            data = test(model, testing)
        else:
            data = pd.concat([data, test(model, testing)])

    data = data.sort("Filter Score")
    n = len(data)

    return [1.0 * sum(data["Soft Call"][i:]) / (n - i) for i in range(n)][::-1]


def train_test_split(events, train_perc):
    """
    Takes in a set of events, and splits it into a training and a testing
    split, where the training set consists of train_perc*100% of the data.
    """

    return events[: int(len(events) * train_perc)], events[
        int(len(events) * train_perc) :
    ]


def threshold_scan(train, test):
    """
    When training is done, a threshold is set on the filter score of the events
    used for training. This will scan a range of these scores, and give the
    accuracy as to each one.
    """

    model, _ = load_yahmm_model("untrained_hmm.txt", False)

    # Get the filter scores for each event in the training set
    train_data = analyze_events(train, model)

    # Store the accuracies in a list
    accuracies = []

    # Scan through a range of thresholds...
    for threshold in 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9:
        print("Threshold set at {}".format(threshold))

        # Take only the events whose filter socre is above a threshold
        event_subset = [
            event for event, score in zip(train, train_data.Score) if score > threshold
        ]

        # If no events left, do not perform training.
        if len(event_subset) == 0:
            continue

        # Open a fresh copy of the HMM.
        model, _ = load_yahmm_model("untrained_hmm.txt", False)

        # Train the model on the training events
        model.fit(torch.tensor(event_subset))

        # Now score each of the testing events
        data = analyze_events(test, model).sort("Score")
        n = len(data)

        # Attach the list of accuracies using the top i events to the frame
        accuracies.append(
            [1.0 * sum(data["Soft Call"][i:]) / (n - i) for i in range(n)][::-1]
        )

    # Turn this list into a numpy array for downstream use
    accuracies = np.array(accuracies)

    return accuracies


def get_events(*args, **kwargs):
    raise Exception("Umimplemented")


if __name__ == "__main__":
    # List all the files which will be used in the analysis.
    files = [
        "14418004-s04.abf",
        "14418005-s04.abf",
        "14418006-s04.abf",
        "14418007-s04.abf",
        "14418008-s04.abf",
        "14418009-s04.abf",
        "14418010-s04.abf",
        "14418011-s04.abf",
        "14418012-s04.abf",
        "14418013-s04.abf",
        "14418014-s04.abf",
        "14418015-s04.abf",
        "14418016-s04.abf",
    ]

    print("Beginning")
    # print("Building Profile HMM...")

    model, state_names = load_yahmm_model("untrained_hmm.txt")

    # Get all the events
    events = get_events(files, model)
    print("{} events detected".format(len(events)))

    train_fold, test_fold = train_test_split(events, train_perc=0.7)
    print(
        "{} Training Events and {} Testing Events".format(
            len(train_fold), len(test_fold)
        )
    )

    model = train(model, train_fold, threshold=0.1)
    data = test(model, test_fold)

    data.to_csv("Test Set.csv")

    import random

    # Cross Validation
    accuracies = []
    for i in range(10):
        print("\nIteration {}".format(i))
        random.shuffle(events)
        accuracies.append(n_fold_cross_validation(events, n=5))

    accuracies = np.array(accuracies)
    np.savetxt("n_fold_accuracies.txt", accuracies)

    plt.plot(accuracies.mean(axis=0))
    plt.show()

    sys.exit()
