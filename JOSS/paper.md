---
title: 'Spectra-Scope : '
tags:
  - Python
  - Materials Science 
  - XANES 
authors:
  - name: Amalya C. Johnson 
    orcid: 0000-0000-0000-0000
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Chris Fajardo 
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Leena Sansguiri 
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 2
  - name: Weike Ye (should Weike be second actually?)
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 2
  - name : Steven B. Torrisi 
    corresponding : true 
    affiliation : 2 
affiliations:
 - name: Materials Science & Engineernig, Stanford University 
   index: 1
   ror: 00hx57361
 - name: Toyota Research Institute 
   index: 2

date: 28 October 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Summary 

# Statement of need

Statement of need 


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements



# References