#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# -----------------------------------------------------------------------------
# 
# Implementation of a two-dimensional DNF-SOM Classifier.
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
from DNFSOM2D import *

# MatPlotLib Options.
plt.rcParams.update({
    "xtick.labelsize"   : 18,
    "ytick.labelsize"   : 18,
    "legend.fontsize"   : 14,
    "axes.titlesize"    : 16,
    "axes.labelsize"    : 14,
    "figure.titlesize"  : 16
})

class DNFSOMClassifier2D(DNFSOM2D):
    def __init__(self, **kwargs):
        """
        Instantiates a DNFSOM2DClassifier object.
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
            np.linalg.norm(self.W[:, np.newaxis] - X, axis = 2),
            axis = 1
        )
        # Assign the respective label to each neuron.
        self.neuron_classes = {
            (i // self.n, i % self.n) : y[idx]
            for i, idx in enumerate(indices)
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
            # Process stimulus for the Dynamic Neural Field.
            input_ = self.process_stimulus(stimulus)
            # Initialise neural activity.
            self.initialise_activity()
            # Simulate neural activation.
            _ = self.simulate(input_)
            # Read-out location.
            bmu_idx = np.argmax(self.V)
            bmu_coords = np.unravel_index(bmu_idx, (self.n, self.n))
            # Predict the corresponding class.
            y_pred.append(self.neuron_classes[bmu_coords])
        return y_pred

    def display_classification(
        self,
        labels  = None,
        figsize = (6, 6),
        ax      = None
    ):
        """
        Displays the learned neural classification.

        Parameters:
            labels  : list or None, optional (default = None).
                A list containing the label names, i.e., how the model
                should refer to each class.
            figsize : tuple, optional (default = (6, 6)).
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
        class_grid = np.full((self.n, self.n), -1)
        for (row, col), class_label in self.neuron_classes.items():
            class_grid[row, col] = class_indices[class_label]
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin = 0, vmax = len(unique_classes) - 1)
        im = ax.imshow(
            class_grid, cmap = cmap, norm = norm, interpolation = "nearest"
        )

        # Define ticks.
        ax.set_title("Learned 2-D Classification")
    
        # Add a colour bar.
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
        sm.set_array([])

        # Display figure, if necessary.
        if show_fig:
            cbar = plt.colorbar(
                sm, ax = ax, ticks = range(len(unique_classes))
            )
            if labels is not None:
                cbar.set_ticklabels([
                    labels[cls] for cls in unique_classes
                ])
            plt.show()

        return im

    def display_prediction(self, X, y, idx, figsize = (12, 6), labels = None):
        """
        Displays the prediction of a DNF-SOM in response to a sample.

        Parameters:
            X       : np.ndarray.
                Set of samples to evaluate.
            y       : np.ndarray.
                True labels of the samples.
            idx     : int.
                Index of the sample to be evaluated.
            figsize : tuple, optional (default = (12, 6)).
                Size of the figure to be displayed.
            labels  : list or None, optional (default = None).
                A list containing the label names, i.e., how the model should
                refer to each class. If None, numerical labels are used.
        """
        # Classification.
        prediction = self.predict(X[idx])[0]
        real_value = y[idx]

        # Access label information, if possible.
        pred_val = labels[prediction] if labels is not None else prediction
        real_val = labels[real_value] if labels is not None else real_value

        # Print predicted vs real class.
        print(f"Pred = {pred_val}")
        print(f"Real = {real_val}")

        # Create figure.
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left: Neural activation.
        im1 = self.display_activity(ax = axes[0])

        # Right: Classification.
        sm2 = self.display_classification(ax = axes[1], labels = labels)

        # Ensure both plots have the same aspect ratio and limits.
        axes[0].set_aspect("equal")
        axes[1].set_aspect("equal")

        axes[0].set_xlim(axes[1].get_xlim())
        axes[0].set_ylim(axes[1].get_ylim())

        # Create colorbar.
        fig.colorbar(
            im1,
            ax          = axes[0],
            label       = "Activation",
            fraction    = 0.046,
            pad         = 0.04
        )
        # Show the figure.
        if sm2 is not None:
            cbar = fig.colorbar(
                sm2, ax = axes[1], fraction = 0.046, pad = 0.04
            )
            if labels is not None:
                cbar.set_ticks(range(len(labels)))
                cbar.set_ticklabels(list(labels))
            cbar.set_label("Classification")
        
        # Ensure subplots have equal size.
        plt.tight_layout()
        plt.show()