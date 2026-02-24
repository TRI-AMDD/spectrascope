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
    equal-contrib: false
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

Spectroscopy is a central class of characterization methodology, providing
useful information on properties
like structure, composition, or excited state dynamics of a given
system. However, many spectroscopic techniques present challenges in development of
interpretable, performant, and reliable supervised learning models due to the wide range of possible
nonlinear correlations that can exist between signal and the response variable (target) of interest.
 Here, we present Spectra-Scope, an open-source AutoML framework implemented in
Python and with a web app for automatic characterization of materials 
properties from spectral data using interpretable machine learning models. It comprises tools for spectral preprocessing, nonlinear feature extraction, machine
learning model training, and feature downselection. Users can easily train
different types of simple, interpretable ML models on a set of feature
transformations quickly and with modest computational resources. In
this work, we outline the methods of Spectra-Scope and its
effectiveness across diverse datasets, with applications to materials
and agricultural spectral data. We show that Spectra-Scope can reproduce
performance of comparable models in the literature, and highlight how our
emphasis on interpretability
can be used to rationalize the behavior of individual models.

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
that seamlessly integrates existing methods for
automated feature generation, model training, 
feature downselection, and inference for spectroscopy data.
Exposing these functionalities in a common package and interface
might help accelerate the model development and data interrogation process for both
experts and novice users.


# Introduction

## Background and Motivation

Spectroscopy is a powerful method for scientific analysis that probes
the interaction of matter with electromagnetic radiation. It allows for
the investigation of electronic, structural, and dynamic properties of
physical systems, and is widely employed across disciplines such as
materials science, chemistry, physics, and the life sciences. Autonomous and high throughput laboratories are increasingly
popular for exploring the synthesis process space for materials
[@szymanski_autonomous_2023; @gregoire_combinatorial_2023; @baird_building_2023; @ebrahimi_advancing_2024; @chao_cuttingedge].
These high-throughput experimental schemes necessitate
companion analysis pipelines for characterization of the complex
datasets they produce.

Automatic spectroscopic analysis tools have thus become a key
component of the high-throughput experimentation pipeline. This analysis
can span from simple peak fitting [@takeuchi_datamanagement] to advanced
machine learning (ML)
[@ogunlade_2024; @solis_mltblg; @SHEREMETYEVA2020455] depending on the
experimental technique and ultimate goal. Spectroscopic models 
can enable real-time
feedback that supports decision-making during experiments
[@liang_realtime], enhance the detection of subtle features and patterns
in data[@zhang_identifying_2020], and alleviate the burden of repetitive
manual analysis [@joy_simple_2022]. Thus, designing robust,
interpretable, and computationally efficient models for spectroscopic
data analysis is a ubiquitous and important challenge in developing high-throughput
experimentation workflows.


# Sections we need to add:
## State of the field - As part of intro

(((PROPOSED OUTLINE HERE)))
- Review some Domain-specific AutoML methods - move from where currently exists
- Shout out general time series and manufacturing AutoML e.g. SPA and Tsfresh
- Deep learning aproaches used
- No-code web interface for computational novices

## Software design- Before Methods

- Focus on lightweight 'glass box' models that can be ran on laptops or via simple web interfaces
- Modular featurization, application that allows mix & match combination of features
- App exposure of functionality makes it easier for experimentalists to work with
- Use of off-the-shelf methods eg SKlearn and SPA 
- Standard visualization libraries used too

## Research Impact Statement- Before conlusion
- Talk about internal usage within Toyota possibly?
- Benchmarking against literature e.g. our reproducibility detailed in the case studies
- Open sourcing and web app
- Documentation
- Notebooks with figure generation explicitly highlighted

## AI Usage Disclosure
(Elsewhere, below)

## Overview of Work

This work describes Spectra-Scope, an open-source AutoML framework with a Python and web app implementation
 that automates simple supervised learning model development from spectral data. It
comprises three main capabilities:
(i) A library of spectral featurizers that can help transform raw spectra into features that better correlate with target properties, (ii) Model training using random forests[@breiman_random_2001] and regularized linear regression, both of which offer handles for interpretability,
and (iii) Demonstrated support for different simultaneous modalities of data. 
The capabilities demonstrated here all only require modest computational resources.
The overall workflow is described in (\autoref{fig:1}).
Full details are provided in the Methods section.

![**Outline of this paper and the Spectra-Scope pipeline.** (a)
Input data can come from any experimental or simulated 1-D array data
source for inference on a scalar response variable. (b) Available
featurizations of spectral data include the cumulative distribution
function, gaussian peak fitting, principal component analysis,
polynomial peak fitting, and others as outlined in the methods. (c)
Transformed spectra are used to train a machine learning algorithm. This
paper focuses on the LCEN algorithm and random forests, but
Spectra-Scope can be incorporated with user built models as well. (d)
Model training of LCEN and random forests includes feature selection
either in the form of LCEN feature downselection or RF feature
importances. 
(e) The algorithms are used to
predict the input response variables. Feature selection helps with model
interpretability and investigating modality importance.\label{fig:1}](Figures/TOC.pdf){width=100%}


# Methods

In this section, we will walk through three key components of our work.
We describe (i) the featurization methods we use, (ii) the models that
we implement, and (iii) an overview of the application.

## Featurization

We present a bundled implementation of "featurizers" to
transform input spectra, extracting features that
may better correlate with a property of interest than the raw spectra.
This compiles techniques seen across the literature in one place.
This allows practitioners to screen over featurizations with no *a priori* knowledge of the
expected relationships between spectral signal and response variables.
Examples include transformations such as the cumulative distribution
function [@chen_robust_2024] and multi-scale polynomial fitting
[@torrisi_random_2020]. 

We categorize our transformations and features into three conceptual sets: 1.
"Local" features (that capture information in finite neighborhoods along
the spectrum e.g. polynomial fitting[@torrisi_random_2020]), 2.
"Nonlocal" features (which incorporate information from the entire
spectrum e.g. Fourier transformations), and 3. "Setwise" features, which
incorporate information from all spectra in the dataset (e.g. PCA).

Local features are useful when spectral features are localized to
specific energy regions. They allow for easy interpretation between
transformed features and energy regions, which can help identify why
specific features may be important for prediction. However, they can
miss global information on e.g. periodic or oscillatory trends that nonlocal
features can highlight. For this reason we provide both local and
nonlocal featurization techniques to complement each other for
prediction. Setwise features like PCA create greater risks for
extrapolation as they rely on properties of the original training
dataset, but may help with understanding properties of the dataset.

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


### Nonlinear Feature Expansion
Because raw spectral values and featurized values may not always have a 
linear correlation with the response variable, nonlinear feature expansion
can help augment the input variables with a set of nonlinear transformations.
For example,
$$x^2, \ln{x}, (\ln{x})^2, \sqrt{x}, \frac{1}{\sqrt{x}}, \frac{1}{x}, \frac{1}{x^2}, x^{1.5}, \frac{\ln{x}}{x}, \frac{1}{\sqrt{x}}$$.
Any feature, such as polynomial coefficients, may be used as inputs to this feature expansion,
but here we focus exclusively on raw 'pointwise' spectral values.
((((QUESTION:: THIS IS TRUE RIGHT?))))
In deployment, any nonlinear transformations used by the model should be vetted to ensure that
overfitting to spurious correlation is not occuring.

## Feature Discovery and Down-Selection

There are several objectives in spectral featurization. The first
 is to find features which have a useful
mathematical correlation with the output of interest, which should have 
 a conceptual connection to the physical mechanism that generates
 the spectrum. Transformed features, by exploring a wider set of data representations,
 may be more likely to improve performance by working better with a model's functional form, such as
 by making the relationship between signal and response linear.
Put differently, transformations may make more similar/dissimilar spectrum-property pairs closer/further in
feature space which makes for easier model fitting.
 One model-free way to identify such relationships 
 is linear and nonlinear correlation analysis[@sun_smart_2021],
using tools such as Pearson's correlation coefficient
[@bertsekas2008introduction] and maximal correlation analysis
[@renyi_measures_1959]. While these model-free methods exist, here, we rely on model performance to judge
feature suitability. Further nonlinear transformations or combinations of features can
be applied as well [@sun_alven_2020-1], which we explore in the Case Study section below.

The second goal is interpretability. There are two primary use cases
here: gaining knowledge about the system, and gaining knowledge about
the model. If the user does not have an *a priori* physical model for the processes
that yield the response variable and/or if the response variable has a
corresponding signal in the spectrum, it may be unknown
which transformations are best. In this case, feature
interpretability may help the user build a mental model for the underlying
process and learn about the system - for example, identifying
if the presence/absence of a feature in the spectrum predicts
the response variable. Further, interpretability can help with model
trustworthiness, as it can help the practitioner diagnose why a failure
during training or inference occurs, or predict possible failures ahead of time.




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

Random forests have many desirable properties:
(1) They can capture nonlinear relationships without intermediate transformations, 
making them useful for 'first-pass' modeling and screening different featurizations.
(2)Random forests train an ensemble of decision trees on
bootstrap aggregates (bagging) of the training dataset, and the trained
trees only use a random subset of the features for each feature split.
This reduces correlation between trees in the ensemble
and mitigates overfitting. For Spectra-Scope it means less negative impact
 when adding irrelevant features, assisting feature screening.
(3) Finally, off-the-shelf implementations of random forests allow users to
gauge the importance of different parts of the feature space
in the training dataset using feature importance scores.
Further interpretability via Shapley additive explanation (SHAP) analysis is possible, but is not implemented here[@lundberg2017unified; @lundberg2018consistent].

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

### Fused LASSO 
We include the fused LASSO as a built-in model to Spectra-Scope. The fused lasso is lasso regression with penalty on the differences of adjacent coefficients [@tibshirani_solution_2011], which is well suited to spectral data. The loss function is 

\begin{equation}
\min_{\beta \in {R}^p} \;
\frac{1}{2}\|y - X\beta\|_2^2 + \lambda \|D\beta\|_1,
\qquad
D =
\begin{bmatrix} 
    -1 & 1 & 0  & \dots & 0 \\
    0 & -1 & 1 \ddots & \vdots \\
    \vdots & \ddots & \ddots & \ddots & 0 \\
    0 & \dots & 0 & -1 & 1
\end{bmatrix}
\in {R}^{(p-1)\times p}.
\end{equation}

It has previously been used to create sparse, interpretable models for battery cycle life prediction and identify regions of interest for closer study[@rhyu_systematic_2025]. 
 Minimizing the difference between adjacent coefficients lends itself to spectroscopy or time-series data because neighboring time or wavelength components are often highly correlated. This can guide partitioning of the input data into smaller sections and improve interpretability. 
 Fused lasso models themselves may be useful, or the regions they identify can be used to build understanding of the relationships between signal and response in a dataset, as we show later in the Case Study section.

## Application

![**Front page of Spectra-Scope application.** Multiple data
types can be input and visualized on the home page. The app includes abilities to 
featurize data, visualize featurizations, train models using random forests or LCEN, 
and visualize the important or downselected features by the model.\label{fig:2}](Figures/Fig2_120925.pdf){width=50%}

Spectra-Scope is available as a no-code web-based application at [spectrascope.matr.io](spectrascope.matr.io){.uri} (See \autoref{fig:2}) and can be hosted locally using the repository.
After data preparation into an appropriate tabular format is done by the
user, uploading data allows for feature generation and model fitting using RF and LCEN.
(((??? QUESTION: ADD A LITTLE MORE DETAIL ABOUT IMPLEMENTATION??? )))

# Case Studies

We benchmark Spectra-Scope on a two distinct spectral datasets from the literature, 
recapitulating results from other studies and demonstrating how the workflow can
be used to build understanding of data.

## XANES + PDF for Transition Metal Oxides

The first dataset consists of transition metal XAS spectra from the
Materials Project
[@jain2013commentary; @mathew2018high; @zheng_automated_2018], and
computed pair distribution functions (PDFs) by Na Narong *et
al.*[@na_narong_interpretable_2025]. Briefly, PDFs describe 
pairwise atomic distance probabilities, encoding local structural information[@TAKESHI201255].
XAS is an element- and orbital-specific measurement X-ray photon absorption.
Many atomic-scale properties can be inferred from XAS
data, including oxidation states, density of states, coordination, and
bond length in a local environment [@chantler_x-ray_2024]. X-ray
absorption near-edge spectra (XANES) refers to the spectra $\sim$ 50-100
eV above the core electron energy level [@chantler_x-ray_2024]. Simulated and experimental
XANES spectra have previously been used to predict oxidation states,
bond length [@chen_robust_2024], Bader charge [@torrisi_random_2020],
and coordination number
[@torrisi_random_2020; @zheng_automated_2018; @na_narong_interpretable_2025]
using machine learning models. For this case, we regress bond length
from XANES and PDF data using random forests and LCEN in Spectra-Scope.


Multi-modal machine
learning has proven a promising way to improve the prediction of certain
materials properties using foundation models[@moro_multimodal_2025].
Combining features from different data sources may be used to both exploit and reveal how different techniques provide complementary information[@na_narong_interpretable_2025]. We highlight Spectra-Scope's ability to compare the performance of
different spectra in prediction, and work with multiple modalities
simultaneously to perform prediction.


### Results

We first regress mean Ti atom nearest-neighbor distance
from simulated XANES spectra of Titanium oxide structures, as was
performed in Na Narong, *et al.* and Torrisi, *et al*
[@na_narong_interpretable_2025; @torrisi_random_2020]. We train both
random forests and LCEN models on the featurized XANES and PDF datasets.
5-fold cross-validation is used to find the optimal model for all
feature and model combinations. Full details of the model training can
be found in the supplementary information. The test root-mean-square-errors (RMSEs)
ranged from 0.024 to 0.073 Å, i.e., percentage errors in the range
1.23 - 3.67 % of the mean bond lengths. 
\autoref{fig:3}a provides a summary
of this regression task. 

Across the board, RF models perform better than the LCEN
models, and some transformation schemes give slight improvement
in performance over using the raw XANES intensity data. The top three
performing features for RF are: the first 10 principal components of the XANES
spectra, polynomial transformations of the
combined XANES and PDF data, and the raw
intensity of the combined XANES and PDF data.
The top featurization schemes for the LCEN models are the
intensity of the combined XANES and PDF data, nonlinear expansions of
the XANES spectra (XANES, NLTrans), and polynomial transformations of
the combined XANES and PDF data. 

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
(e), or both XANES and PDF (f) datasets. The most
important polynomial features selected from the XANES dataset come from
a similar energy region as the CDF, near the main edge, further
highlighting its importance in regressing bond length. When combining
the two datasets, the most important features come from regions across
the datasets, rather than concentrated in particular energetic regions (in agreement with Ref. [@na_narong_interpretable_2025]).

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

Next, Spectra-Scope is applied to a publicly available experimental dataset of
optical spectra from wine grapes to predict their pH and sugar content by
Ebrahimi *et al* [@ebrahimi_advancing_2024]. The differences of 
this dataset to the transition metal oxide dataset highlights the flexibility of Spectra-Scope.

The dataset included Visible-Near-Infrared (Vis-NIR) and Raman spectra
of two types of grape varieties, and ground truth labels of the acidity
and total soluble solids (TSS) content of grapes. Understanding the
chemical composition of grapes helps winemakers make
informed decisions during growth and harvesting. Sugar content and
acidity are important indicators for determining when to harvest grapes
and can influence the potential alcoholic content and fermentation
process of wines, respectively [@cramer_transcriptomic_2014]. Spectral analysis 
allows non-invasive, real time measurements of grape chemistry
at vineyards, reducing the need for costly off-site processing and analysis
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
to predict TSS using the Vis-NIR and Raman spectra from the dataset. We use similar preprocessing steps as Ebrahimi *et al.*: centering and scaling the data, and restricting the Vis-NIR spectra to between 400 and 1300 nm, where the strongest absorption values are. The
models are trained on the raw data or features generated by different
transformations. Again, we can use Spectra-Scope to quickly visualize
model performance of different models and feature transformations as
given in \autoref{fig:4}a. Our reported % RMSE are similar or better than that reported by Ebrahimi, *et al.* [@ebrahimi_advancing_2024] for similar models. \autoref{fig:4}a suggests that a
better prediction of TSS is found when training on the Vis-NIR
absorption spectra.

Thus, we focus primarily on Vis-NIR spectra for our analysis.
In \autoref{fig:4}b and c we highlight the
20 most important features or highest magnitude coefficients using (i) interpolated data and (ii) polynomial coefficients for both the (b) Random Forest and (c) LCEN models. Each featurization and model combination highlights different components of the spectra as important for prediction. In \autoref{fig:4}b(i), the random forest selects many wavelengths from 450-900 as important. In (ii), the polynomial coefficients around 738nm, 970nm, and ~1200nm are important. In \autoref{fig:4}c(i), the LCEN model selects many high magnitude coefficients around 970 nm and 1200nm. In (ii), polynomial coefficients around 738nm, 840nm, 970nm, 1100nm, and 1200nm are selected by the model. 

Many of the wavelengths selected as important by the models correspond with second or third overtones of molecular infrared vibrational frequencies that may be prevalent in grapes. The absorption peak at 970nm likely corresponds with water, which has a near-IR absorption band near 970 nm due to overtones of the O-H stretching mode of water [@lin_theory_2009; @golic_short-wavelength_2003]. The third overtone of this vibrational frequency is at 738nm, which is picked up by 3 of the model-feature pairs. The small feature around 840nm may correspond with the second overtone of the 1100nm O-H combination band in water. The broader peak at 1200nm may correspond with the second overtone of C-H and C-H$_2$ stretching frequencies, which would come from organic solutes like glucose and sucrose, and have been reported at 1100-1230nm and 1215nm, respectively [@golic_short-wavelength_2003]. These spectral features have previously been used for glucose monitoring [@golic_short-wavelength_2003]. For the interpolated and polynomial featurization schemes shown in \autoref{fig:4}c, the final LCEN model selects 86 and 33 features, respectively.
((???? QUESTION: USING WHICH REGULARIZATION PARAMETER????))
((???? QUESTION: SHOULD WE INCLUDE THIS FOR XANES TOO????))

 Random forest models  predict on all the available features in a dataset, which are of size 899 and 157, respectively. This highlights the utility of regularization for feature down-selection, as nonzero features can be more easily judged for their importance than a feature score which is computed for all features in a random forest model.

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



\autoref{fig:5} shows the regression coefficients of a fused lasso model trained on the raw intensity data. As the fused lasso penalizes differences between neighboring coefficients, the model returns regions of the spectra with the same or very similar coefficients. Positive/negative coefficient values suggest that data within these regions correlate/anticorrelate with the output. As the regularization term increases, regions that are less important for prediction will approach 0. The regions that remain "on" with greater regularization are ~550nm to ~700nm and ~700nm to ~850nm. This suggests that the 738nm overtone that was highlighted by the Random Forest and LCEN models in \autoref{fig:4}, is important for this case as well. Additionally, the coefficients change value near 840nm, 970nm, and 1150nm for all regularization parameters. This highlights these wavelengths as important, as they resist the regularization penalty imposed by the fused lasso model.
The fact the most consequential (judging by largest coefficient magnitude) regions correlate with the vibrational modes of organic solutes indicated above builds confidence in this model.
Seeing model agreement across different featurization modes in certain regions can be taken as evidence that they are fitting on consistent signal.

![**LCEN and Fused Lasso selected Features.** The top panel shows all of the NIR spectra in the dataset. The bottom panel shows the regression coefficients for fused lasso models with different regularization parameters $\alpha$.\label{fig:5}](Figures/Fig5_121525.pdf)


Finally, we brieflycompare feature-performance correlations across datasets. As a sanity check, we find that the features that perform the best with the LCEN model (and overall) have the highest pearson's correlation coefficients in this dataset. This suggests that LCEN models perform well due to nonlinear transformations uncovering linear correlations with the target variable. For the XANES + PDF dataset, there is less of a correlation between those features with the highest pearson's correlation coefficients, and the lowest RMSE. This suggests that the underlying relationship is more nonlinear, and could explain why the random forests perform better than LCEN for this case study. 

![**Linear Correlation Assessment.** Analysis for (a) XANES + PDF and (b) Grapes dataset for predicting their respective target variable. r : pearson's correlation coefficient. Corresponding metric for each model. \label{fig:6}](Figures/Correlation_121525.pdf){width=80%}


# Discussion

These case studies highlight how the ability to compare a variety of transformations and models at once allows a Spectra-Scope user to quickly identify which model may be best for their regression or classification task.
Our agreement with past literature validates that Spectra-Scope recovers physically meaningful features, supporting the toolkit's reliability for exploratory analysis.

Attempting different featurizations and model types and comparing which parts of spectra are used may be used for hypothesis formation and building confidence. For example, if fused LASSO and polynomial featurization highlight the same regions, it may be taken as evidence that the energy ranges contain meaningful signal.
When using LCEN, the subsets of data that are used by the fitting procedure should be studied as the regularization parameter changes.
We see consistency in our fused lasso results, but we encourage checking this in future deployment.

There are some limitations to the AutoML approach. 
As mentioned earlier, using model performance alone as a  "winner-take-all" measure of feature selection risks overfitting.
 Models that perform slightly worse across the same set of validation tests may work better in deployment or generalize better to out-of-distribution data if they verifiably are fitting on physically meaningful signals.
Sanity checks on the feature sets employed should be performed for physical plausibility.
For example, if there is expected to be periodic signal in the data, then the fourier and wavelet transforms are more sensible choices of featurization.
Depending on dataset size, it may be more desirable to perform first-pass maximal correlation analysis. We use the `ace` package in our library to perform this, but do not discuss this in the manuscript.

Here, we only explore random forests and LCEN. Poor performance with these models doesn't mean that a data-driven approach should be disregarded. Maximum accuracy may come from more dedicated feature and model design.
If more data is available, featurization techniques may not be as necessary, as one of the hallmarks of artificial neural networks is that they learn their own representations of data depending on the task at hand. They may in these cases obtain better performance and robustness.
We expect featurization techniques to be most useful in a lower-data regime (e.g. order 100s-1000s of data points), where they represent a form of inductive bias introduced to the data.

Furthermore, some use cases may be beyond the scope of Spectra-Scope.
Uncertainty quantification is not formally supported.
For multimodality, we demonstrate simple concatenation of feature sets together for simultaneous study, but expect that
more sophisticated methods that combine information from two different datasets could be employed.
((( QUESTION: CITE EXACTNN? OTHER STUFF?? )))



We highlight some failure modes for models discovered using an AutoML approach.
Users should verify that adequately performing feature sets correspond with meaningful physical signals and the needs of future deployment.
For example, identifying correlations between the Vis-NIR regions of interest and vibrational modes that are likely to matter for $^\circ$Brix data.
If regions highlighted as important are known to have low signal-to-noise ratios, then the model should be vetted for overfitting, as spurious correlations may be picked up by the model.
For setwise features like PCA, understanding the similarity between inference-time data and training data is very important.




# Conclusion

Spectra-Scope is a flexible platform for building interpretable machine learning models with spectral data. 
We highlight its ability to draw insights from diverse datasets and aid practitioners in better
understanding their data and models. Its accessibility is enhanced by its corresponding web app, allowing it to be used by scientists with and without Python experience. 
While we've currently focused on spectral
data, Spectra-Scope can take any array-like data as input, allowing it to be used with other characterization techniques, and applied to
problems beyond materials science.

We hope that a focus on sparsity and interpretability will make for
easier deployment of models in production in contexts ranging from automated materials discovery and high-throughput experimentation to industrial manufacturing. For example, models which operate on only
a few features may be more easy to verify, troubleshoot, and deploy to
operate without supervision. For future users, Spectra-Scope is an open
source Python package available at
[github.com/TRI-AMDD/spectrascope](github.com/TRI-AMDD/spectrascope){.uri}
and a web application available at
[spectrascope.matr.io](spectrascope.matr.io){.uri}.

# Acknowledgements
The authors thank Pedro Seber for advice on the implementation of LCEN,
Richard Braatz, Tina Na Narong, and Simon Billinge for helpful
discussions, and Linda Hung for helpful feedback on the manuscript.

# AI Usage Disclosure
Grammar, concision, and typo checks were performed using Claude Opus 4.5. Generated suggestions were considered but not necessarily applied, 
and all changes were directly mediated after review by the authors. All authors have reviewed and approved all information demonstrated in this work.

# Supplementary Information 

## NEEDS TO BE WRITTEN WITH THE INFO AS PROMISED ABOVE

# References