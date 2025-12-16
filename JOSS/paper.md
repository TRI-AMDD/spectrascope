---
title: 'Spectra-Scope : A toolkit for automatic characterization of materials properties from spectral data'
tags:
  - Python
  - Materials Science 
  - Spectroscopy
  - Materials characterization 
  - X-Ray absorption 
  - Auto-ML 
authors:
  - name: Amalya C. Johnson 
    orcid: 0000-0000-0000-0000
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Chris Fajardo 
    equal-contrib: false›
    affiliation: 1
  - name: Leena Sansguiri 
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Weike Ye 
    orcid: 
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 1
  - name : Steven B. Torrisi 
    orcid: 
    corresponding : true 
    affiliation : 1 
affiliations:
 - name: Toyota Research Institute 
   index: 1

date: 9 December 2025
bibliography: paper.bib

---

# Summary

Spectroscopy is a pillar of materials science. It can provide
crucial information to experimentalists and theorists on physical properties
like the structure, composition, or excited state dynamics of a
system. However, many spectroscopic techniques feature challenges like
complicated modeling needed for analysis, low signal-to-noise ratio
for specific materials, or expensive equipment needed for high
resolution studies. Here, we present Spectra-Scope, an open-source
Python package and web app for automatic characterization of materials 
properties from spectral data using interpretable machine learning models. It
supports spectral preprocessing, nonlinear feature extraction, machine
learning model training, and feature selection. Users can easily train
different types of simple, interpretable ML models on a set of feature
transformations quickly and with modest computational resources. In
this work, we outline the methods of Spectra-Scope and its
effectiveness across diverse datasets, with applications to materials
and agricultural datasets. 

# Statement of need

Automated machine learning (Auto-ML) software pipelines
streamline and automate common ML tasks like data preparation, feature
engineering, model training, and hyper parameter tuning. 
While Auto-ML is a well-established technique in machine learning communities,
few tools exist to apply and optimize such pipelines for
materials-specific data. Previous works have applied Auto-ML techniques 
to specific materials systems or experimental pipelines, but lack generalizability to other
experimental techniques or materials datasets [@tsamardinos_automated_2020; @ji_research_2022]. 
More general purpose frameworks have been established for manufacturing or 
time-series data [@sun_smart_2021; @sun_alven_2020; @tsfresh]. 
To the best of our knowledge, the literature lacks a versatile toolkit 
that seamlessly integrates automated feature generation, model training, 
feature downselection, and inference for spectroscopy data. 

# Introduction

## Background and Motivation

Spectroscopy is a powerful method for scientific analysis that probes
the interaction of matter with electromagnetic radiation. It allows for
the investigation of electronic, structural, and dynamic properties of
physical systems, and is widely employed across disciplines such as
materials science, chemistry, physics, and the life sciences. Recently,
spectroscopic techniques have become integrated into high-throughput
experimentation schemes and autonomous laboratories, which are
increasingly used in materials science for accelerated exploration and
development of materials
[@szymanski_autonomous_2023; @gregoire_combinatorial_2023; @baird_building_2023; @ebrahimi_advancing_2024; @chao_cuttingedge].
However, these high-throughput experimental schemes necessitate
companion analysis pipelines for characterization of the complex
datasets they produce.

Automatic spectroscopic analysis tools have thus become a critical
component of the high-throughput experimentation pipeline. This analysis
can span from simple peak fitting [@takeuchi_datamanagement] to advanced
machine learning (ML)
[@ogunlade_2024; @solis_mltblg; @SHEREMETYEVA2020455] depending on the
experimental technique and ultimate goal, and can enable real-time
feedback that supports decision-making during experiments
[@liang_realtime], enhance the detection of subtle features and patterns
in data[@zhang_identifying_2020], and alleviate the burden of repetitive
manual analysis [@joy_simple_2022]. Thus, designing robust,
interpretable, and computationally efficient models for spectroscopic
data analysis is a key challenge and opportunity in high-throughput
experimentation and analysis. Analytic pipelines that are simple,
reliable, and highlight physical insights in spectral data can help
advance automated experimentation and characterization, thereby
accelerating materials discovery and development.

## Overview of Work

Here, we present Spectra-Scope, a toolkit for the development of
characterization models for materials properties based on feature
exploration and interpretable machine learning models. This
general-purpose framework is built for feature generation, model training, and 
feature downselection from spectral data (\autoref{fig:1}). A main component
of Spectra-Scope is a a set of implemented transformations, or
"featurizers" to transform input spectra and extract new features that
may better correlate with a property of interest than the raw spectra.
For example, transformations such as the cumulative distribution
function [@chen_robust_2024] and multi-scale polynomial fitting
[@torrisi_random_2020] have proven successful for prediction of
coordination number and mean nearest-neighbor distance, respectively,
from XAS spectra of transition metal oxide structures. The available
featurizers in Spectra-Scope allow a practitioner to explore what
transformations may be best for their data.

The generated features are then used to train a machine learning
algorithm to predict a property of interest. These features are model
agnostic, and can be implemented into any machine learning algorithm.
However, the user interface and toolkit we implemented here is primarily
built around random forests (RF) and regularized linear regression
models. Due to their simplicity and interpretability, these models allow
users to identify the most important features for prediction via the
feature importances returned by random forests [@breiman_random_2001]
and feature downselection during regularized linear regression. Further,
we incorporate the LCEN algorithm [@seber_lcen_2024-1] which combines
lasso and elastic net regression with "clip" steps that provide
additional constraints for feature downselection. Further details on
featurization, feature downselection, and the available models are
discussed in the Methods section. These
techniques allow Spectra-Scope to highlight the most important features
of a spectrum for prediction of a specific property within a particular
dataset. This provides users with "glass box" models where one can
identify how the model obtains answers and weed out erroneous fits. More
importantly, it can elucidate physical relationships between spectra and
materials properties if specific peaks, energies, or spectral features
are labeled as important predictors for a given property.

Finally, Spectra-Scope has the ability to train models on combined
spectra from different experimental techniques. Multi-modal machine
learning has proven a promising way to improve the prediction of certain
materials properties using foundation models[@moro_multimodal_2025].
This may be useful if different techniques provide information that,
when combined, aids a model in prediction. Feature selection can
highlight how spectral features of different techniques are related and
transformations may be able to uncover relationships previously
unidentified in the raw data [@torrisi_random_2020; @na_narong_interpretable_2025].

We hope that a focus on sparse and interpretable models will make for
easier application of the featurization strategies highlighted by
Spectra-Scope into production. For example, models which operate on only
a few features may be more easy to verify, troubleshoot, and deploy to
operate without supervision. For future users, Spectra-Scope is an open
source Python package available at
[github.com/TRI-AMDD/spectrascope](github.com/TRI-AMDD/spectrascope){.uri}
and a web application available at
[spectrascope.matr.io](spectrascope.matr.io){.uri}.

![**Outline of this paper and the Spectra-Scope pipeline.** (a)
Input data can come from any experimental or simulated 1-D array data
source for inference on a scalar response variable. (b) Available
featurizations of spectral data include the cumulative distribution
function, gaussian peak fitting, principal component analysis,
polynomial peak fitting, and others as outlined in the methods. (c)
Transformed spectra are used to train a machine learning algorithm. This
paper focuses on the LCEN algorithm and random forests, but
Spectra-Scope can be incorporated with user built models as well. (d)
Model training of LCEN and random forests will include feature selection
either in the form of feature downselection with LCEN, or feature
importances with random forests. This allows for the identification of
important features for prediction. (e) The algorithms are used to
predict the input response variables. Feature selection helps with model
interpretability and investigating modality importance.\label{fig:1}](Figures/TOC.pdf){width=100%}


# Methods

In this section, we will walk through three key components of our work.
We describe (i) the featurization methods we use, (ii) the models that
we implement, and (iii) an overview of the application.

## Featurization

We categorize our transformations and features into three sets: 1.
"Local" features (that capture information in finite neighborhoods along
the spectrum e.g. polynomial fitting[@torrisi_random_2020]), 2.
"Nonlocal" features (which incorporate information from the entire
spectrum e.g. Fourier transformations), and 3. "Setwise" features, which
incorporate information from all spectra in the dataset (e.g. PCA).

Local features are useful when spectral features are localized to
specific energy regions. They allow for easy interpretation between
transformed features and energy regions, which can help identify why
specific features may be important for prediction. However, they can
miss global information on periodic or oscillatory trends that nonlocal
features can highlight. For this reason we provide both local and
nonlocal featurization techniques to complement each other for
prediction. Setwise features like PCA create greater risks for
extrapolation as they rely on properties of the original training
dataset.

We summarize our featurization methods in the table below. Additional information 
on the different featurizations can be found in the [API documentation](github.com/TRI-AMDD/spectrascope){.uri}. 

  **Featurization**                  **Type**
  ---------------------------------- ----------
  Interpolation                      \-
  Multiscale Polynomial Fitting      Local
  Gaussian Peak Fitting              Local
  Continuous Wavelet Transform       Local
  Nonlinear expansion                Local
  Fourier Transform                  Nonlocal
  Cumulative Distribution Function   Nonlocal
  Principal Component Analysis       Setwise
  Feature Agglomeration              Setwise

## Feature Discovery and Down-Selection

Spectral data featurization has several objectives. The first practical
goal is to find transformed spectral features which provide a useful
mathematical correlation with the output of interest, which may also
have a conceptual connection to the physical mechanism that generates
the spectrum. Transformed features may then be more likely to work well
with a model's functional form (such as linear regression in LCEN or
nonlinear regression by random forests) and therefore improve
performance. Put differently, the transformations may have the effect of
making similar/dissimilar spectrum-property pairs closer/further in
feature space which then makes it easier for models to be fit to said
relationships. One way to identify such a relationship without fitting a
model is linear and nonlinear correlation analysis[@sun_smart_2021],
using tools such as Pearson's correlation coefficient
[@bertsekas2008introduction] and maximal correlation analysis
[@renyi_measures_1959]. Here, we rely on model performance to determine
goodness of fit, which assumes that the model is capable of capturing
the relationships between input and output variables present in the
data. Further nonlinear transformations or combinations of features can
be applied as well [@sun_alven_2020-1].

The second goal is for interpretability. There are two primary use cases
here: gaining knowledge about the system, and gaining knowledge about
the model. If the user does not have a physical model for the processes
that yield the response variable, and/or if the process has a
corresponding expression in the spectrum, they may not know *a priori*
which transformations will be best. In this case, feature
interpretability may help the user build a model for the underlying
process and gain knowledge about the system - for example, identifying
if the presence or absence of a feature in the spectrum is predictive of
the response variable. Secondly, interpretability can help with model
trustworthiness, as it can help the practitioner diagnose why a failure
during training or inference occurs.

## Models

The next step is to train models to learn the relationship between the
spectra (and/or its transformed form) and a property of interest. We
note as a caveat that Auto-ML approaches suffer from a
"winner-take-all\" approach [@sun_smart_2021] which means that a model
that is selected based on performance may have limitations that actually
make it suboptimal in practice. For example, trying every possible model
increases the degrees of freedom in training and can lead to overfitting
[@arlot_survey2009]. Here, we hope that the emphasis on sparse and
interpretable models helps mitigate the risk of overfitting by making
clear what features the model is predicting on. In practice, we
encourage users in production to examine the models to ensure that the
features of interest correlate with the underlying phenomenon.

### Random forests

We refer readers who are unfamiliar with random forests to
Breiman, 2001 [@breiman_random_2001]. We use the implementation from Scikit-Learn[@scikit-learn].

Random forests have many desirable properties, but we highlight the
three most relevant ones for this work: First, random forests can
capture nonlinear relationships between input and response variables
without intermediate transformations. By contrast, linear models perform
best when the features are transformed such that the correlation between
the input features and output is linear. This means that random forests
are easier to use for 'first-pass' modeling efforts and therefore might
more easily help users discover useful featurization techniques.
Secondly, random forests train an ensemble of decision trees on
bootstrap aggregates (bagging) of the training dataset, and the trained
trees only use a random subset of the features for each feature split.
This has many desirable results, like reducing correlation between trees
and mitigating overfitting. For our purposes, it means a negative impact
on model performance is less severe when adding irrelevant features.
Therefore, random forests can handle a large feature space and so can be
used when attempting to discover useful featurization strategies.
Finally, off-the-shelf implementations of random forests allow users to
access a measure of importance of different parts of the feature space
in the training dataset. When random forests are used in Spectra-Scope,
the user is able to easily identify and visualize the most important
features in model training, making the model more interpretable and
possibly highlighting physical relationships between a spectral feature
and property. Further interpretability with random forests can be
obtained using Shapley additive explanation (SHAP) analysis, but we do
not implement this in this work [@lundberg2017unified; @lundberg2018consistent].

### LCEN

The Lasso-Clip-Elastic-Net (LCEN) algorithm of Seber and Braatz,
introduced and described in detail elsewhere, [@seber_lcen_2024] works
by sequential fitting and 'clipping' of linear regression coefficients
under different regularization conditions.

We briefly summarize the four main steps of the LCEN fitting procedure.
First, LASSO regression [@Tibshirani-1996] is performed on the input features. Second, there is a clip step, where coefficients with absolute magnitude below a user selected cutoff are set to 0. Third, elastic net regression is performed on the features that have not been clipped in step 2. Finally, a second clip step is performed on the
resultant coefficients. The clip steps perform feature downselection to
create accurate but sparse models. The algorithm can also perform
transformations on the input data to find nonlinear relationships
between the data and the label. Including these transformations can be
turned on or off through Spectra-Scope. Here, we use the implementation
of LCEN from the Smart Process Analytics (SPA) repository
[@seber_github].

The clip steps, by eliminating coefficients of small magnitude, mean
that any surviving nonzero coefficient is making a significant
contribution to the prediction. When the features are scaled, the
magnitude of the coefficients can also be used to interpret the relative
contributions. Therefore, interpretability comes from examining the
magnitudes of the coefficients and the user can learn more about which
features are contributing the most.

### Fused Lasso 
We include the fused lasso as a built-in model to Spectra-Scope. The fused lasso is lasso regression with penalty on the differences of adjacent coefficients [@tibshirani_solution_2011]. The loss function is 

\begin{equation}
\min_{\beta \in {R}^p} \;
\frac{1}{2}\|y - X\beta\|_2^2 + \lambda \|D\beta\|_1,
\qquad
D =
\begin{bmatrix} 
    -1 & 1 & 0  & \dots & 0 \\
    0 & -1 & 1 * \ddots & \vdots \\
    \vdots & \ddots & \ddots & \ddots & 0 \\
    0 & \dots & 0 & -1 & 1
\end{bmatrix}
\in {R}^{(p-1)\times p}.
\end{equation}

It has previously been used to create sparse, interpretable models for battery cycle life prediction [@rhyu_systematic_2025]. Minimizing the difference between adjacent coefficients lends itself to spectroscopy or time-series data because neighboring time or wavelength components are often highly correlated. This can guide partitioning of the input data into smaller sections and improve interpretability. 

## Application

![**Front page of Spectra-Scope application.** Multiple data
types can be input and visualized on the home page. The app includes abilities to 
featurize data, visualize featurizations, train models using random forests or LCEN, 
and visualize the important or downselected features by the model.\label{fig:2}](Figures/Fig2_120925.pdf){width=50%}

In addition to an open-source Python library, Spectra-Scope is available
as a web-based application for convenient no-code usage. Data
preparation into an appropriate tabular format must first be done by the
user, but then can be used as input for the Spectra-Scope workflow.
\autoref{fig:2} shows the landing page for the app. 
While more flexibility is availability when
interfacing with Spectra-Scope with Python, the app provides easily
accessible model comparisons and feature downselection for quick
analyses. The app is available at
[spectrascope.matr.io](spectrascope.matr.io){.uri}.

# Case Studies

We benchmark Spectra-Scope on a variety of spectral datasets with
diverse origins, modalities, and underlying physical characteristics.
These test cases highlight the flexibility of Spectra-Scope, its ability
to highlight physical relationships between spectra and properties, and
its ease of application to a variety of contexts.

## XANES + PDF for Transition Metal Oxides

The first dataset consists of transition metal XAS spectra from the
Materials Project
[@jain2013commentary; @mathew2018high; @zheng_automated_2018], and
computed pair distribution functions (PDFs) by Na Narong *et
al.*[@na_narong_interpretable_2025]. Briefly, PDFs describe the
probability of finding a pair of atoms separated by a given distance in
a material, encoding local structural information[@TAKESHI201255], and
XAS measures the absorption of X-ray photons in a material. In XAS,
absorption is dominated by core-level electrons and is element and
orbital-specific. Many atomic-scale properties can be inferred from XAS
data, including oxidation states, density of states, coordination, and
bond length in a local environment [@chantler_x-ray_2024]. X-ray
absorption near-edge spectra (XANES) refers to the spectra $\sim$ 50-100
eV above the core electron energy level and involves multiple scattering
of the photoelectron [@chantler_x-ray_2024]. Simulated and experimental
XANES spectra have previously been used to predict oxidation states,
bond length [@chen_robust_2024], Bader charge [@torrisi_random_2020],
and coordination number
[@torrisi_random_2020; @zheng_automated_2018; @na_narong_interpretable_2025]
using machine learning models. For this case, we regress bond length
from XANES and PDF data using random forests and LCEN in Spectra-Scope.
We highlight Spectra-Scope's ability to compare the performance of
different spectra in prediction, and work with multiple modalities
simultaneously to perform prediction.

### Results

First, we use Spectra-Scope to regress mean nearest-neighbor distance
from simulated XANES spectra of Titanium oxide structures, as was
performed in Na Narong, *et al.* and Torrisi, *et al*
[@na_narong_interpretable_2025; @torrisi_random_2020]. We train both
random forests and LCEN models on the featurized XANES and PDF datasets.
5-fold cross-validation is used to find the optimal model for all
feature and model combinations. Full details of the model training can
be found in the supplementary. The test root-mean-square-errors (RMSEs)
ranged from 0.024 to 0.073 Å, i.e., percentage errors in the range
1.23 - 3.67 % of the mean bond lengths. Figure
\autoref{fig:3}a provides a summary
of this regression task. We find that the random forest models perform
the best, and that some transformation schemes give slight improvement
in performance over using the raw XANES intensity data. The top three
performing features are: the first 10 principal components of the XANES
spectra, polynomial transformations of the
combined XANES and PDF data, and the raw
intensity of the combined XANES and PDF data.

Across the board, the random forest models perform better than the LCEN
models. The top featurization schemes for the LCEN models are the
intensity of the combined XANES and PDF data, nonlinear expansions of
the XANES spectra (XANES, NLTrans), and polynomial transformations of
the combined XANES and PDF data. The nonlinear feature expansion
augments the input variable x with a set of nonlinear transformations:
including
$$x^2, \ln{x}, (\ln{x})^2, \sqrt{x}, \frac{1}{\sqrt{x}}, \frac{1}{x}, \frac{1}{x^2}, x^{1.5}, \frac{\ln{x}}{x}, \frac{1}{\sqrt{x}}$$.
As LCEN is a linear model, the
performance of the nonlinear transformations and the polynomial
transformations over the raw XANES intensity highlights that there is a
nonlinear relationship between the spectra and the target that the model
is trying to uncover. The better performance of the random forest models
over the LCEN models further supports this, as random forests are more
easily able to uncover nonlinear relationships.

\autoref{fig:3}(b-f) gives
visualizations for a few of the different transformations used for this
regression task. The cumulative distribution function (CDF) has
previously been shown to give good performance on regressing bond length
for simulated and experimental XANES data of Ni oxide structures
[@chen_robust_2024]. \autoref{fig:3}b shows the average XANES spectra from the dataset in
black, and all of the CDF of the spectra in blue. The vertical dashed
lines give the top three most important features for prediction, which
occur near the main edge of the spectra where there is the strongest
absorption.

The lowest RMSE for this case comes from training a random forest on the
10 principal components of the XANES spectra. \autoref{fig:3}c visualizes the
first two principal components of the data, and the corresponding bond
length. \autoref{fig:3}d
highlights the top 10 (d,e) and top 20 (f) important polynomial
coefficients identified for prediction using just the XANES (d), PDF
(e), or both XANES and PDF (f) datasets. The polynomial coefficients
featurization scheme extracts the shape of the spectra and can be
sensitive to peak-like features in data. When combining the two
datasets, this featurization scheme may help identify shapes in the
different spectra that may work together for prediction. The most
important polynomial features selected from the XANES dataset come from
a similar energy region as the CDF, near the main edge, further
highlighting its importance in regressing bond length. When combining
the two datasets, the most important features come from regions across
the datasets, rather than concentrated in particular energetic regions.

Our models perform similar to previous studies using random forests to
regress bond length of transition metal oxide structures. Na Narong, *et
al.* [@na_narong_interpretable_2025] report percentage errors of
3.1-3.9% of the mean bond lengths using the intensity data of XANES,
PDFs, and combined spectra for the same structures. Torrisi, *et al.*
[@torrisi_random_2020] report an $R^2$ score of 0.85 for the same
regression task using polynomial transformations of XANES spectra. For
the same featurization scheme, we measure an $R^2$ of 0.84. Chen, *et
al.* [@chen_robust_2024] perform bond length regression on Ni-Oxide
structures using different featurization schemes for XANES spectra. They
report test RMSE of 0.009, 0.011, and 0.020 for the raw intensity, CDF,
and PCA featurizations, respectively, using random forests for
regression. These compare with our values of 0.029, 0.028, and 0.024 for
the same featurizations for Ti-oxide structures. We attribute
differences in model performance to the different datasets used across the studies. 

![**Regressing mean nearest-neighbor distance from simulated
XANES spectra and PDFs of Ti-oxide structures.** (a) Summary of RMSE for
regressing bond length using LCEN and random forests for XANES, PDF,
XANES + PDF, and other transformations of the data. CDF : cumulative
distribution function. NLTrans : Nonlinear feature expansion as outlined
in the main text and the supplementary. Clustering : Feature
agglomeration clustering. The top three features and model combinations
correspond with green, blue, and red stars, respectively. Comparison of
the important features identified when using (b) the CDF transformation
of XANES spectra, (c) the first 10 principal components of XANES
spectra, and polynomial transformations of (d) XANES, (e) PDF, and (f)
XANES + PDF for regression with random forests. (b) CDF of all XANES
spectra in blue. Vertical dashed line : top three important features for
prediction using the CDF. (c) First two principal components of the
XANES spectra colored by bond length. (d) All XANES spectra from the
dataset (blue), average spectrum (black) and corresponding 10 most
important features for prediction (characteristic polynomial images).
The top three most important features are highlighted by the vertical
dashed line. (e) All PDF from the dataset (red), average spectrum
(black), and positions of top 10 most important features for prediction.
The top three most important features are highlighted by the vertical
dashed line. (f) Average XANES spectra (blue) and PDF (red) for
simultaneously using both datasets for regression. Dashed lines show the
top three most important features extracted from fitting XANES (blue)
and PDF (red) spectra separately.\label{fig:3}](Figures/fig3_120825.pdf)

## Raman + NIR for Wine Grapes

Next, Spectra-Scope is applied to a publicly available dataset of
optical spectra of wine grapes to predict their pH and sugar content by
Ebrahimi *et al* [@ebrahimi_advancing_2024]. The difference in
technique, properties, and materials of this dataset to the transition
metal oxide dataset highlights the flexibility of Spectra-Scope.

The dataset included Visible-Near-Infrared (Vis-NIR) and Raman spectra
of two types of grape varieties, and ground truth labels of the acidity
and total soluble solids (TSS) content of grapes. Understanding the
chemical composition of grapes is essential for winemakers to make
informed decisions during growth and harvesting. Both sugar content and
acidity are important indicators for determining when to harvest grapes
and can influence the potential alcoholic content and fermentation
process of wines, respectively [@cramer_transcriptomic_2014]. These
methodologies allow non-invasive, real time analysis of grape chemistry
at vineyards, and is an improvement over the typical process of sending
grapes to outside laboratories to determine their composition
[@ebrahimi_advancing_2024]. Vis-NIR spectra have been applied to
estimate TSS in a variety of crops
[@jha_nondestructive_2004; @Liu_2011], and Raman spectroscopy is used to
identify molecular profiles present in a sample.

### Results

Ebrahimi *et al.* compare the performance of different machine learning
algorithms on predicting pH and TSS in two grape varieties when using
either the full spectrum or the spectrum transformed using principal
component analysis with 6 or 15 components [@ebrahimi_advancing_2024].
They found a root mean squared error (RMSE) of 5-7% when predicting TSS
of French grapes. We use this dataset to compare the performance of
different transformations and models using Spectra-Scope. The % RMSE is defined as
$100 \times \frac{\text{RMSE}}{{y_{\max}}}$ where $y_{\max}$ is the
maximum value of the $^\circ$Brix data.

We train a series of random forest and LCEN models using Spectra-Scope
to predict TSS using the Vis-NIR and Raman spectra from the dataset. We use similar preprocessing steps as Ebrahimi *et al.* : centering and scaling the data, and restricting the Vis-NIR spectra to between 400 and 1300 nm, where the strongest absorption values are. The
models are trained on the raw data or features generated by different
transformations. Again, we can use Spectra-Scope to quickly visualize
model performance of different models and feature transformations as
given in \autoref{fig:4}a. Our reported % RMSE are similar or better than that reported by Ebrahimi, *et al.* [@ebrahimi_advancing_2024] for similar models. \autoref{fig:4}a suggests that a
better prediction of TSS is found when training on the Vis-NIR
absorption spectra.

In \autoref{fig:4}b and c we highlight the
20 most important features or highest magnitude coefficients using the interpolated data (i) and the polynomial coefficients (ii) featurization scheme for both the Random Forest (b) and LCEN (c) models. Each featurization and model combination highlights different components of the spectra as important for prediction. In \autoref{fig:4}b(i), the random forest selects many wavelengths from 450-900 as important. In (ii), the polynomial coefficients around 738nm, 970nm, and ~1200nm are important. In \autoref{fig:4}c(i), the LCEN model selects many high magnitude coefficients  around 970 nm and 1200nm. In (ii), polynomial coefficients around 738nm, 840nm, 970nm, 1100nm, and 1200nm are selected by the model. 

Many of the wavelengths selected as important by the models correspond with second or third overtones of molecular infrared vibrational frequencies that may be prevalent in grapes. The absorption peak at 970nm likely corresponds with water, which has a near-IR absorption band near 970 nm due to overtones of the O-H stretching mode of water [@lin_theory_2009; @golic_short-wavelength_2003]. The third overtone of this vibrational frequency is at 738nm, which is picked up by 3 of the model-feature pairs. The small feature around 840nm may correspond with the second overtone of the 1100nm O-H combination band in water. The broader peak at 1200nm may correspond with the second overtone of C-H and C-H$_2$ stretching frequencies, which would come from organic solutes like glucose and sucrose, and have been reported at 1100-1230nm and 1215nm, respectively [@golic_short-wavelength_2003]. These spectral features have previously been used for glucose monitoring [@golic_short-wavelength_2003]. For the interpolated and polynomial featurization schemes shown in \autoref{fig:4}c, the final LCEN model selects 86 and 33 features, respectively. This is in contrast to the random forest models which will predict on all the available features in a dataset, which are 899 and 157, respectively. This highlights the power of a sparse and simple
model for prediction in this dataset, and the significance of the regions selected for prediction. 

![**Regressing grape sugar content.** (a) % RMSE for random
forests and LCEN models built on Vis-NIR and Raman spectra transformed
in different ways. The top three features and model combinations correspond with green, blue, and
red stars, respectively. (b) Top 20 most important features for predicting TSS
with the full spectrum (i) and polynomial features extracted from the
spectrum (ii) using random forests. (c) 20 highest absolute magnitude
coefficients for regressing TSS with the full spectrum (i) and
polynomial features extracted from the spectrum (ii) using LCEN. Blue
vertical lines : selected/important features. Characteristic polynomials
: selected/important features.\label{fig:4}](Figures/fig4_121525.pdf)

\autoref{fig:5} shows the regression coefficients of a fused lasso model trained on the raw intensity data. As the fused lasso penalizes differences between neighboring coefficients, the model returns regions of the spectra with the same or very similar coefficients. This suggests that data within these regions give similar information to predicting the output. As the regularization term increases, regions that are less important for prediction will be sent to 0. The regions that remain "on" with greater regularization are ~550nm to ~700nm and ~700nm to ~850nm. This suggests that the 738nm overtone that was highlighted by the Random Forest and LCEN models in \autoref{fig:4}, is important for this case as well. Additionally, the coefficients change value near 840nm, 970nm, and 1150nm for all regularization parameters. This highlights these wavelengths as important, as they resist the regularization penalty imposed by the fused lasso model.  

![**LCEN and Fused Lasso selected Features.** The top panel shows all of the NIR spectra in the dataset. The bottom panel shows the regression coefficients for fused lasso models with different regularization parameters $\alpha$.\label{fig:5}](Figures/Fig5_121525.pdf)


# Discussion

These case studies highlight how the ability to compare a variety of transformations and models at once allows a Spectra-Scope user to quickly identify which model may be best for their regression or classification task. While training many models can lead to over-fitting, and identifying the model with the lowest RMSE from this summary may not be the best model to generalize to other materials or spectra, this procedure can be useful for getting a quick summary on the performance of simple tasks. 

Further, both datasets show the importance of feature downselection in model interpretation. In both cases, similar parts of spectra are highlighted as important across different featurization schemes, suggesting that different models are identifying truly physically
important regions. This can help identify erroneous fits if some featurization schemes select random or unphysical regions of spectra for
prediction.

Finally, for both cases, non-linear transformations of features perform better than just using the raw, interpolated spectra as the training
data. Using Spectra-Scope, we can quickly analyze the linear correlation and maximal correlation between the generated features and the output. We compute 
Pearson's correlation coefficient using 
`sklearn.feature_selection.r_regression`. Maximal correlation analysis is implemented with the `ace` package in Spectra-Scope, but is not explored in this text. 

For the grapes dataset, we find that the features that perform the best with the LCEN model (and overall) have the highest pearson's correlation coefficients in this dataset. This suggests that LCEN models perform well due to the nonlinear transformations uncovering linear correlations with the target variable. For the XANES + PDF dataset, there is less of a correlation between those features with the highest pearson's correlation coefficients, and the lowest RMSE. This suggests that the underlying relationship is more nonlinear, and could be why the random forests perform better than LCEN for this case study. 

![**Linear Correlation Assessment.** Analysis for (a) XANES + PDF and (b) Grapes dataset for predicting their respective target variable. r : pearson's correlation coefficient. Corresponding metric for each model. \label{fig:6}](Figures/Correlation_121525.pdf){width=80%}

# Conclusion

Spectra-Scope is a flexible platform for building interpretable machine learning models with spectral data. Here we highlight its ability to draw insights from diverse datasets and aid practitioners in better
understanding their data and models. Its accessibility is enhanced by its corresponding web app, allowing it to be used by scientists with and without Python experience. While we've currently focused on spectral
data, Spectra-Scope can take any array-like data as input, allowing it to be used with other characterization techniques, and applied to
problems beyond materials science.

Machine learning has emerged as a promising tool for accelerating materials discovery, however highly flexible black box models like neural networks need large datasets, can have high variance, and can
obscure physical relationships between data and a prediction. To contrast, Spectra-Scope is well-positioned for aiding materials
scientists due to its ability to produce sparse and generalizable models with LCEN, work well with small experimental datasets, and identify
important spectral features for prediction. This toolkit may be beneficial for applications from autonomous materials discovery to
high-throughput experimentation to industrial manufacturing.

# Acknowledgements
The authors thank Pedro Seber for advice on the implementation of LCEN,
Richard Braatz, Tina Na Narong, and Simon Billinge for helpful
discussions, and Linda Hung for helpful feedback on the manuscript.

# Supplementary Information 
Spectra-Scope is an open-source Python package available at
github.comTRI-AMDDspectrascope and a web application available at
spectrascope.matr.io.

# References