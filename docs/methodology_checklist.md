- [ ] Are all baselines (zero-shot, local-only) evaluated on the same n
      as Fed-ICL?
- [ ] Is the held-out test set evaluated on the same n?
- [ ] Is the random seed fixed and recorded?
- [ ] Is the Dirichlet partition reproducible from the seed?
- [ ] Has the result file been renamed before the next run?
- [ ] Has the run's purpose been logged in the lab notebook?

The single most important check is the first one: comparing
differently-sized baselines will nearly hid a real federation gain. Easy to miss; hard to detect after the fact.