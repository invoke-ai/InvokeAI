# Workflow accordion resource measurements

The accordion adds the `collapsible` form container element (`type: 'collapsible'`, `data: { content: string; children: string[] }`), saved expansion state, hidden-field validation, and layout-aware boundaries. Only the activated Workflow widget loads this code; Launchpad and inactive widget checks remain enforced, and non-workflow route script budgets remain unchanged.

## Upstream baseline merge (`02e3ee0e5b44c90c1be4d959025449c56f27d6a7`)

`performance/architecture-baseline.json` is kept identical to `origin/main` (`02e3ee0e5b44c90c1be4d959025449c56f27d6a7`). In `performance/browser-baseline.json`, all upstream route timings, runner metadata, and non-workflow route baselines from `origin/main` are preserved unchanged (the +221 locale bytes in `en.json` stay well within the 4,096-byte route allowance); only the `editor-workflow` route (`empty` and `representative`) is updated for the lazy Workflow chunk delta (+7,658 script bytes, +221 locale bytes in `resourceBaseline.otherRawBytes`) and its two source owners:

| Activated metric (`editor-workflow`) | Upstream `origin/main` baseline | Updated with `collapsible` container |
| ------------------------------------ | ------------------------------: | -----------------------------------: |
| Largest asset, raw bytes             |                         210,757 |                              218,415 |
| Script raw bytes                     |                         459,316 |                              466,974 |
| Total raw bytes                      |                         474,729 |                              482,387 |
| Requests                             |                               6 |                                    6 |
| Script requests                      |                               5 |                                    5 |

The only added source owners in `browser-baseline.json` are `src/features/workflow/ui/linear/formSections.ts` and `src/features/workflow/ui/linear/formSectionStatus.ts` inside the `editor-workflow-static` owner set. Other fixture owner sets and request counts are unchanged.
