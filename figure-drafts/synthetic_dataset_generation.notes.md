# Synthetic dataset generation figure

Included in the controlled experimental design section of `paper/main.tex`; the
experiment configuration table is in the training-protocol appendix.

- `synthetic_dataset_generation.pdf`: vector PDF, 200 × 130.28 mm, embedded fonts.
- `synthetic_dataset_generation.svg`: editable vector source with live text.
- `synthetic_dataset_generation.png`: review preview.
- `make_synthetic_dataset_generation.py`: reproducible renderer using the existing data generators.

The figure depicts the two original experiments described in `paper/main.tex`. It does not depict the later expanded sweep,
whose sampling budgets and holdouts differ. Blue denotes attribute-defined
relations; red denotes random controls. Filled and hollow circles distinguish
relation-trained and relation-held-out subjects in A. Dashed links show held-out
relations, which are not present in the training corpus.

All subject split markers, illustrative random assignments, attribute bits, and
relation matrices are generated with seed 19. A shows every one of the 192
subjects and 32 objects; its mapping example expands the six subjects in f₀.
The concrete training sentence uses a₃₂, a trained subject. B uses readable
excerpts: six real entities (s₀, s₁, s₅, s₂, s₃, s₄) and their ten attribute bits,
plus the corresponding 6 × 6 submatrices of the full 64 × 64 pair-label tables.
Rows are subjects and columns are objects in the relation tables. All three
tables use the same entity order, grouped by attribute 0; r₀ is the example
equality table. The highlighted pair (s₀, s₁) has attribute 0 = 1 for both entities,
giving equality label y but random label n. This same pair is repeated in the
training sentences. Colored cells mean y; pale cells mean n, also written directly
in every visible cell. The small excerpt need not retain the full tables' per-row
class balance. The four-node edge-switch
glyph is a schematic of degree-preserving randomization, not a depiction of the
full 64-node graph. The random table has 31 non-self neighbors plus its self-pair.
Binary prompt schemas omit the initial `<bos>` token for readability.

A's F = 16 corpus contains 28,672 attribute sentences and 2,048 sentences per
relation (32,768 total). B's equal-width blue and red bar halves denote the total
equality and random relation sampling mass during joint training. Each of the ten
blue segments is one equality relation; the red relation receives ten times that
mass. Attribute supervision continues throughout the equality warm-up and joint
training; the first 6,000 steps omit the random relation. Class-dependent sampling/loss weights
are shared by the structured and random relations and omitted from this diagram.

Suggested caption: **Generating two controlled synthetic relation datasets.**
**A**, Subjects span a Cartesian grid of categorical attributes. A fixed cyclic
attribute shift defines a many-to-one object relation, paired with an independent
balanced random assignment. Both relations share subject splits and exact
repetition counts; held-out subjects retain attribute supervision. **B**, Unique,
balanced binary attribute vectors define ten equality relations. Degree-preserving
graph randomization followed by self-loops produces a matched random relation.
The tables share symmetry and per-entity class balance. Token examples and the
joint-stage relation sampling mixture show how each construction supplies training
data. Seed-19 worlds are shown; B uses the same six-entity excerpt in all three
tables and traces the contrasting labels of one concrete pair into sentences.

Visual inspiration: direct entity-to-entity arrows and aligned comparison panels
in [Hernandez et al., ICLR 2024, Fig. 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/2b6d7285306fe896fca9b2e9a02ff6ec-Paper-Conference.pdf),
and the concrete corpus-to-relation overview in
[Merullo et al., ICLR 2025, Fig. 1](https://proceedings.iclr.cc/paper_files/paper/2025/file/b35c6f99df526a798433fabff039bd50-Paper-Conference.pdf).
The artwork is original; no source figure is reproduced.

Regenerate from the repository root:

```sh
experiments/.venv/bin/python figure-drafts/make_synthetic_dataset_generation.py
```

The renderer checks balanced target counts in both A splits, balanced and unique
binary attributes, full attribute rank, and all B tables' symmetry, diagonals,
and row degrees before exporting. No model training or manuscript build is run.
