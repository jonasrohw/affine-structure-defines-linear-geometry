# NAACL 2027 template provenance

NAACL 2027 main-conference submissions follow ACL Rolling Review (ARR) requirements
and use the shared ACL template, rather than a separate year-specific style.

Official sources checked on 2026-09-28:

- [NAACL 2027 main-conference call](https://2027.naacl.org/calls/main_conference_papers/)
- [ARR paper submission requirements and template](https://aclrollingreview.org/cfp#paper-submission-process-criteria-and-template)
- [ACL style repository](https://github.com/acl-org/acl-style-files)
- [ACL formatting guidelines](https://acl-org.github.io/ACLPUB/formatting.html)

The unmodified `acl.sty` and `acl_natbib.bst` files are vendored from upstream commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409`. Their SHA-256 digests are:

```text
19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de  acl.sty
6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e  acl_natbib.bst
```

`main.tex` uses the official 11-point article class and `\usepackage[review]{acl}`
for anonymous, two-column review with line numbers. Bibliographies use
`acl_natbib`. The system TeX installation supplies standard packages, including
`natbib`. Wide figures and tables span both columns; appendix text remains in
two columns. The existing limitations have their own section after the conclusion.

Long-paper submissions allow eight pages of main content, plus limitations,
references, and appendices under the ARR rules. Accepted long papers receive one
additional content page. The review setting should change to `final` only when
preparing the camera-ready paper.

## Build the paper

From the repository root:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript source is `main.tex`, and the compiled manuscript is `main.pdf`.
The migrated PDF has eight pages of main content through the conclusion;
limitations start on page 9, followed by references and two-column appendices.
