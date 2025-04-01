# DNF-SOM Classifier

This repository provides an implementation of the **Dynamic Neural Field Self-Organizing Map (DNF-SOM)** for classification tasks.

## Repository Structure

The repository is organized as follows:

- **Models** - Implements the base **DNF-SOM** and its classification extension for both **1D and 2D spatial domains**.
- **Examples** - Includes Jupyter Notebooks demonstrating the model on three standard Machine Learning datasets, namely iris, wine, and breast cancer.
- **Poster** - Contains the poster presented at **JOCLAD 2025** (ISCAP, Porto).

## Dynamic Neural Fields

Dynamic Neural Fields (DNFs) [1] are **Recurrent Neural Networks** that describe the coarse-grained activity of populations of interacting neurons **organized in a continuous feature space** $$\Omega \subset \mathbb{R}^q, \, \forall q \in \mathbb{N}_0$$.

**Information** in DNFs **is represented by supra-threshold, localized regions of neural activity**, commonly referred to as bumps.

A common mathematical formulation of DNFs is given by the **integro-differential equation**:

$$\tau\frac{\partial u(x,t)}{\partial t}=-u(x,t)+I(x,t)+\int_{\Omega}w_l(|x-y|)f[u(y,t)]\,dy$$,

where:

- $$u(x,t)$$ : Neural activation at site $$x \in \Omega$$ and time $$t \geq 0$$;
- $$\tau > 0$$: Time constant;
- $$I(x,t)$$: Localized external stimulus;
- $$w_l(x)$$: Synaptic interactions, separated into excitatory and inhibitory, i.e., $$w_l(x)=w_e(x)-w_i(x)$$.
- $$f(x)$$: Firing rate function taken as a rectification, i.e., $$f(x) = \text{max}(x, 0.0)$$.

## Dynamic Neural Field Self-Organizing Maps

Consider a dataset $$\mathcal{D}=\{(x_i, y_i)\}_{i=1}^{N}$$, where $$x_i \in \mathbb{R}^m$$ for $$1 \leq i \leq N$$, i.e., each feature $$x_i$$ has $$m$$ attributes, and $$y_i$$ represents the corresponding label. The DNF-SOM [2] initializes each neuron's codeword to have the same number of attributes, i.e., $$w_f \in \Omega \times \mathbb{R}^m$$.

At each epoch, a random stimulus $s$ is drawn from $$\mathcal{D}$$.

The input for the DNF is given by the following function:

$$I(x, t) = 1.0 - \frac{|w_f(x) - s|_1}{m}.$$

Note: To ensure $$I(x,t) \in [0, 1]$$, the input data must be normalized beforehand.

The neural activity is simulated, and the codebooks are updated according to:

$$\frac{\partial w_f(x,t)}{\partial t} =\gamma \, (s - w_f(x, t)) \, \int_{\Omega} w_e(|x -y|)\text{rect}[u(y, t)] \, dy,$$

where $$\gamma > 0$$ is the learning rate.

## Classification and Prediction

The dataset is partitioned into training and testing sets, denoted as $$\mathcal{D}_{\text{train}} = (x_{\text{train}}, y_{\text{train}})$$ and $$\mathcal{D}_{\text{test}} = (x_{\text{test}}, y_{\text{test}})$$, respectively.

After training, classification and prediction are performed as follows:

### **Classification Algorithm**
**Input:** Training set $$\mathcal{D}_{\text{train}}$$; Codebook $$w_f(x)$$.
**Output:** Assigned labels for each neuron $$x \in \Omega$$.

1. **For** each $$x \in \Omega$$: 
    - Determine the sample that is most similar to the neuron's codebook.
    - Identify the sample that is most similar to the neuron's codebook
    $$\text{idx} \leftarrow \arg\min_{\,i} |w_f(x)-x_{\text{train}, i}|_1$$
    - Assign the corresponding class to the neuron.
    $$\text{classf}[x]\leftarrow y_{\text{train}}[\text{idx}]$$
2. **Return** classf.

### **Prediction Algorithm**
**Input:** Sample $$s \in x_{\text{test}}$$; Codebook $$w_f(x)$$; Neuron classifications classf.  
**Output:** Predicted class $$\hat{y}$$.

1. Compute input for DNF:  
   $$I(x, t) \leftarrow 1.0 - \frac{|w_f(x) - s|_1}{m}.$$
2. Simulate neural dynamics until convergence:  
   $$u(x) \leftarrow \text{Simulate neural dynamics}$$
3. Find the location of the bump:  
   $$\text{idx} \leftarrow \arg\max_{x \in \Omega} u(x)$$
4. Predict class:  
   $$\hat{y} \leftarrow \text{classf[idx]}$$  
5. **Return** $\hat{y}$.

## References

[1] Shun-Ichi Amari. Dynamics of pattern formation in lateral-inhibition type neural fields. Biological cybernetics, 27(2):77–87, 1977

[2] Detorakis, G. I., \& Rougier, N. P. (2014). Structure of receptive fields in a computational model of area 3b of primary sensory cortex. Frontiers in Computational Neuroscience, 8, 76.