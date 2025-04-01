#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# -----------------------------------------------------------------------------
# 
# Implementation of a one-dimensional DNF-SOM Classifier.
# Copyright (C) 2025 Paulo Barbosa.
#
#    This program is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by the
#    Free Software Foundation, either version 3 of the License, or (at your
#    option) any later version.
#
#    This program is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#    or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#    for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------
#
# Third-Party Code Attribution
#
# This project contains code adapted from the SITopMAps repository, authored by
# Georgios Is. Detorakis, which is licensed under the BSD-3-Clause license.
#
# URL to Repository: https://github.com/gdetor/SITopMaps/
#
# -----------------------------------------------------------------------------

# Libraries.
import matplotlib.pyplot as plt

# Dynamic Neural Field Self-Organising Map.
from DNFSOM1D import *

# MatPlotLib Options.
plt.rcParams.update({
    "xtick.labelsize"   : 18,
    "ytick.labelsize"   : 18,
    "legend.fontsize"   : 14,
    "axes.titlesize"    : 16,
    "axes.labelsize"    : 14,
    "figure.titlesize"  : 16
})

class DNFSOMClassifier1D(DNFSOM1D):
    def __init__(self, **kwargs):
        """
        Instantiates a DNFSOM1DClassifier object.
        """
        super().__init__(**kwargs)

    def classify(self, X, y):
        """
        Assigns a class to each neuron in the field.

        Parameters:
            X       : np.ndarray.
                Training set's features.
            y       : np.ndarray, optional.
                Training set's labels.
        """
        # Determine which training instance is closest to each codeword.
        indices = np.argmin(
            np.linalg.norm(self.W[:, np.newaxis] - X, axis = 2), axis = 1
        )
        # Assign the respective label to each neuron.
        self.neuron_classes = {
            i : y[idx] for i, idx in enumerate(indices)
        }
        
    def fit(
        self, 
        X,
        y,
        T       = 100.0,
        dt      = 0.35,
        lrate   = 0.05,
        epochs  = 3000,
        verbose = -1
    ):
        """
        Self-organise the feed-forward weights to the training data.
        Afterwards, it performs classification by assigning a label
        to each neuron in the Dynamic Neural Field.

        Parameters:
            X       : np.ndarray.
                Training set's features.
            y       : np.ndarray.
                Training set's labels.
                Defaults to None, if the dataset is unsupervised.
            T       : float, optional (default = 100.0).
                Total simulation time.
            dt      : float, optional (default = 0.35).
                Time step value.
            lrate   : float, optional (default = 0.05).
                Learning rate for the feed-forward weights.
            epochs  : int, optional (default = 3000).
                Number of sample presentations.
            verbose : int, optional (default = -1).
                Interval to display training progress.
                If -1, then no output is produced.
        """
        # Update the feed-forward weights.
        super().fit(
            X,
            y,
            T       = T,
            dt      = dt,
            lrate   = lrate,
            epochs  = epochs,
            verbose = verbose
        )
        # Perform classification according to the resulting weights.
        self.classify(X, y)

    def predict(self, X):
        """
        Predicts the respective class for each sample.

        Parameters:
            X       : np.ndarray.
                Set of samples to evaluate.
        
        Returns:
            np.ndarray.
                Predicted class for each sample.
        """
        y_pred = []
        for stimulus in np.atleast_2d(X):
            # Process the stimulus for the Dynamic Neural Field.
            input_ = self.process_stimulus(stimulus)
            # Initialise neural activity.
            self.initialise_activity()
            # Simulate neural activation.
            _ = self.simulate(input_)
            # Read-out location.
            location = np.argmax(self.V)
            y_pred.append(self.neuron_classes[location])
        return y_pred

    def display_classification(
        self,
        labels  = None,
        figsize = (6, 2),
        ax      = None
    ):
        """
        Displays the learned neural classification.

        Parameters:
            labels  : list or None, optional (default = None).
                A list containing the label names, i.e., how the model
                should refer to each class.
            figsize : tuple, optional (default = (6, 2)).
                Size of the figure to be displayed.
            ax      : matplotlib.axes.Axes or None, optional (default = None).
                Axes to plot on. If None, a new figure is created.
        """
        # Determine whether to create a new figure.
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
            show_fig = True
        else:
            show_fig = False
    
        # Fetch unique class labels.
        unique_classes = sorted(set(self.neuron_classes.values()))
        class_indices = { cls : i for i, cls in enumerate(unique_classes) }

        # Define colour map.
        cmap = plt.cm.viridis
        sc = ax.scatter(
            list(self.neuron_classes.keys()),
            [0] * len(self.neuron_classes),
            c = [
                class_indices[self.neuron_classes[i]]
                for i in self.neuron_classes
            ],
            cmap = cmap,
            s = 200,
            marker = "s"
        )

        # Define ticks.
        ax.set_xlabel("Neuron Index")
        ax.set_yticks([])
        ax.set_title("Learned 1-D Representation")

        # Display figure, if necessary.
        if show_fig:
            cbar = plt.colorbar(sc, ax = ax, label = "Class Label")
            # Display provided label names, if possible.
            if labels is not None:
                cbar_ticks = np.arange(len(unique_classes))
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(labels)
            plt.show()

        return sc

    def display_prediction(self, X, y, idx, labels = None):
        """
        Displays the prediction of a DNF-SOM in response to a sample.

        Parameters:
            X       : np.ndarray.
                Set of samples to evaluate.
            y       : np.ndarray.
                True labels of the samples.
            idx     : int.
                Index of the sample to be evaluated.
            labels  : list or None, optional (default = None).
                A list containing the label names, i.e., how the model should
                refer to each class. If None, numerical labels are used.
        """
        # Classification.
        prediction = self.predict(X[idx])[0]
        real_value = y[idx]

        # Access label information, if possible.
        pred_val = prediction if labels is None else labels[prediction]
        real_val = real_value if labels is None else labels[real_value]
    
        # Print predicted vs real class.
        print(f"Pred = {pred_val}")
        print(f"Real = {real_val}")

        # Create figure.
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize = (10, 4), sharex = True,
            gridspec_kw = {"height_ratios": [3, 1]}
        )

        # Top: Neural activation.
        self.display_activity(ax = ax1)
        # Bottom: Classification.
        sc = self.display_classification(labels = labels, ax = ax2)

        # Create colorbar.
        if sc is not None:
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(sc, cax = cbar_ax, label = "Class")
            if labels is not None:
                cbar_ticks = np.arange(len(labels))
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels(labels)
    
        # Show the figure.
        plt.show()