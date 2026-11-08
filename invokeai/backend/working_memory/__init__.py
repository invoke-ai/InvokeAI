"""Working-memory estimates as data: what an operation needs on the compute device, and how that was priced.

An estimator turns an operation's runtime facts (`features`) and a set of fitted coefficients into a
`WorkingMemoryEstimate` that `LoadedModel.model_on_device(working_memory=...)` reserves. The coefficients ship as data
under `data/`, next to the measured points that bound them; `tests/backend/working_memory/` enforces both.
"""
