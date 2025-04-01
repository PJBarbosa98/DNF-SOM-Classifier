#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# -----------------------------------------------------------------------------
# 
# Implementation of a two-dimensional Dynamic Neural Field Self-Organising Map.
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
import numpy as np
import matplotlib.pyplot as plt

# Convolution via FFT.
from numpy.fft import rfft2, irfft2, ifftshift

class DNFSOM2D:
    """
    Dynamic Neural Field Self-Organizing Map in Two Spatial Dimensions.
    """
    def __init__(
        self, n_units = 128, tau = 1.0, alpha = 0.1, seed = 42
    ):
        """
        Instantiates a DNF-SOM object.

        Parameters:
            n_units     : int, optional (default = 128).            
                Number of neurons in the population. 
            tau         : float, optional (default = 1.0).
                Time constant of the dynamics.
            alpha       : float, optional (default = 0.1).
                Synaptic scaling.
            seed        : float, optional (default = 42).
                Random seed for reproducibility.
        """
        # Parameters.
        self.n      = n_units
        self.tau    = tau
        self.alpha  = alpha

        # Reproducibility.
        np.random.seed(seed)

        # Initialisation.
        self.initialise_activity()
        self.initialise_synapses()

    def initialise_activity(self):
        """
        Initialises neural activation ramdomly between 0.00 and 0.01.
        """
        self.U = np.random.rand(self.n, self.n) * 0.01
        self.V = np.random.rand(self.n, self.n) * 0.01

    def initialise_fft(self):
        """
        Computes the FFT for faster convolution computation.
        """
        self.We_fft = rfft2(ifftshift(self.We[::-1, ::-1]))
        self.Wi_fft = rfft2(ifftshift(self.Wi[::-1, ::-1]))

    def initialise_synapses(
        self, Ke = 3.72, sigma_e = 0.10, Ki = 2.40, sigma_i = 1.00
    ):
        """
        Initialises recurrent excitatory and inhibitory connections.

        Parameters:
            Ke      : float, optional (default = 3.72).
                Strength of excitation.            
            sigma_e : float, optional (default = 0.10).
                Spread of excitation.
            Ki      : float, optional (default = 2.40).
                Strength of inhibition.
            sigma_i : float, optional (default = 1.00).
                Spread of inhibition.
        """
        # Determine amplitudes.
        ampl_e = 960.0 / (self.n * self.n) * Ke
        ampl_i = 960.0 / (self.n * self.n) * Ki

        # Define the recurrent connections as a difference of Gaussians.
        self.We = ampl_e * self.gaussian(
            (self.n, self.n), (sigma_e, sigma_e)
        ) * self.alpha
        self.Wi = ampl_i * self.gaussian(
            (self.n, self.n), (sigma_i, sigma_i)
        ) * self.alpha

        # Adjust recurrent connections for FFT.
        self.initialise_fft()

    def process_stimulus(self, stimulus):
        """
        Prepares stimulus for Dynamic Neural Field.

        Parameters:
            stimulus    : np.ndarray.
                Stimulus to be processed for self-organisation.

        Returns:
            np.ndarray
                Stimulus in one spatial dimension for the Dynamic Neural Field.
        """
        input_ = 1.0 - np.abs(self.W - stimulus).sum(axis = 1)
        input_ = (input_ - input_.min()) / (input_.max() - input_.min())
        # input_ = input_ * self.alpha
        input_ = input_.reshape(self.n, self.n)
        return input_

    def simulate(self, input_, T = 100.0, dt = 0.35):
        """
        Simulates the neural response to a stimulus.

        Parameters:
            input_  : np.ndarray.
                Stimulus (before processing) to the neural field.
            T       : float, optional (default = 100.0).
                Total simulation time.
            dt      : float, optional (default = 0.35).
                Time step value.

        Returns:
            Le      : np.ndarray.
                Excitatory component of the recurrent connections within the
                Dynamic Neural Field.
        """
        for _ in range(int(T)):
            Z = rfft2(self.V)
            Le = irfft2(Z * self.We_fft, (self.n, self.n)).real
            Li = irfft2(Z * self.Wi_fft, (self.n, self.n)).real
            self.U += (-self.U + (Le - Li) + input_) * dt * self.tau
            self.V = np.maximum(self.U, 0.0)
        return Le

    def display_activity(self, figsize = (6, 6), ax = None):
        """
        Displays the current neural activation.

        Parameters:
            figsize : tuple, optional (default = (6, 6)).
                Size of the figure to be displayed.
            ax      : matplotlib.axes.Axes or None, optional (default = None).
                Axes to plot on. If None, a new figure is created.
        """
        # Determine whether to create a new figure.
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
            plot_fig = True
        else:
            plot_fig = False
        
        # Plot the neural activation.
        im = ax.imshow(self.V, cmap = "viridis", aspect = "auto")
        
        # Define axes labels and title.
        ax.set_title("Dynamic Neural Field Response")
        
        # Show the figure, if specified.
        if plot_fig:
            cbar = plt.colorbar(im, ax = ax)
            cbar.set_label("Activation")
            plt.show()
        
        return im

    def fit(
        self, 
        X,
        y       = None,
        T       = 100.0,
        dt      = 0.35,
        lrate   = 0.05,
        epochs  = 3000,
        verbose = -1
    ):
        """
        Self-organise the feed-forward weights to the training data.

        Parameters:
            X       : np.ndarray.
                Training set's features.
            y       : np.ndarray, optional (default = None).
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
        # Initialise the feed-forward weights.
        self.W = np.random.rand(self.n * self.n, X.shape[-1])
        # Generate stimuli.
        X_samples, _ = self.draw_samples(X, y, epochs)
        # Training loop.
        for epoch in range(epochs):
            # Draw a stimulus.
            stimulus = X_samples[epoch]
            # Process the stimulus for the Dynamic Neural Field.
            input_ = self.process_stimulus(stimulus)
            # Initialise neural activity.
            self.initialise_activity()
            # Simulate neural activation.
            Le = self.simulate(input_, T = T, dt = dt)
            # Update the weights.
            self.W -= lrate * (Le.ravel() * (self.W - stimulus).T).T
            # Display progress as specified.
            if (verbose != -1) and (epoch % verbose == 0):
                print(f"Epoch = {epoch:4d}.")

    @staticmethod
    def gaussian(shape = (25, 25), width = (1, 1), centre = (0, 0)):
        """
        Computes a generalised Gaussian distribution.

        Parameters:
            shape   : tuple, optional (default = (25, 25)).
                Shape of the spatial discretisation.
            width   : tuple, optional (default = (1, 1)).
                Width of the distribution.
            centre  : tuple, optional (default = (0, 0)).
                Mean of the distribution.

        Returns:
            np.ndarray.
                Gaussian distribution in accordance to the parameterisation.
        """
        grid = [slice(0, size) for size in shape]
        C = np.mgrid[tuple(grid)]
        R = np.zeros(shape)
        for i, size in enumerate(shape):
            if size > 1:
                R += (((C[i] / (size-1)) * 2 - 1 - centre[i]) / width[i]) ** 2
        return np.exp(-R / 2)

    @staticmethod
    def draw_samples(X, y = None, n_samples = 1):
        """
        Draws samples from a set.

        Parameters:
            X           : np.ndarray.
                Dataset's features.
            y           : np.ndarray, optional (default = None).
                Dataset's labels.
            n_samples   : int, optional (default = 1).
                Number of samples to generate.

        Returns.
            (np.ndarray, np.ndarray).
                Generated sample's features and labels, respectively.
        """
        indices = np.random.choice(
            X.shape[0], size = n_samples, replace = True
        )
        if y is not None:
            return X[indices], y[indices]
        return X[indices], None