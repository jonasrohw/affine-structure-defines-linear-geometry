# PaperReview.ai iteration

The anonymous manuscript PDF was submitted to PaperReview.ai on 2026-09-01. The
completed review assigned a numerical score of 7.3/10 and marked claims support,
experimental soundness, writing clarity, prior-work context, question importance,
originality, and value to the community positively.

## Writing-guide pass

Before external review, the manuscript was checked against all five requested
writing resources:

- [Jakob Foerster's ML paper guide](https://www.jakobfoerster.com/how-to-ml-paper)
  informed the single-thesis structure and the problem, method, evidence sequence
  in the abstract and introduction.
- [Writing a good scientific paper](https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper)
  informed the first-page summary figure, the introduction funnel, and captions
  that state the inference supported by each plot.
- [How to write a scientific paper](https://users.aalto.fi/~jsaramak/HowToWriteCommsCoffee.pdf)
  informed paragraph-level argument order, early definitions, and the separation
  of results from discussion.
- [Scientific writing](https://web.stanford.edu/~jfries/assets/slides/BIOMEDIN-212_scientific-writing.pdf)
  informed the claim-evidence structure, direct topic sentences, and explicit
  limitations.
- [Writing tips](https://www.cs.ubc.ca/~schmidtm/Courses/Notes/writing.pdf)
  informed terminology consistency, sentence-level compression, and the removal
  of vague qualifiers.

Advice about citation mechanics was not applied when it differed from the
conference rules. Citations and references follow the ICLR 2027 author kit.

The revision incorporated the review's requests that could be answered from the
existing evidence:

- The experimental-design section now explains why Experiment 1 uses held-out
  nearest-object accuracy while Experiment 2 uses overdetermined full-data
  multivariate R-squared.
- The appendix now defines midpoint defect and Jacobian dispersion mathematically
  and states the pair set, diagnostic fold, singular-value threshold, and use of
  exact automatic-differentiation Jacobians.
- The centroid target is motivated against an additive classifier on concatenated
  subject-object rows. The metric discussion also explains why ridge and low-rank
  affine fits cannot improve the full-data residual minimized by OLS, and why a
  bilinear head tests a different representation class.
- The related-work section now distinguishes a post hoc OLS decoder from
  regression-like algorithms executed by transformers and cites controlled work
  that separates decodability from causal use.
- The limitations now state explicitly that the study did not ablate inverse
  attribute examples or search every hidden layer for affine structure.

Suggestions that require new experiments, including a full inverse-supervision
ablation, architecture sweep, and layer-wise probe sweep, were not converted into
claims without evidence. They are recorded as limitations or open tests in the
manuscript. Citation formatting continues to follow the ICLR 2027 author kit.
