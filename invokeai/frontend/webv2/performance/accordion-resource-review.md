# Workflow accordion resource measurements

The accordion adds opt-in heading-based collapsible sections (`data.collapsible`), saved expansion state, hidden-field validation, and layout-aware boundaries. Only the activated Workflow widget loads this code; Launchpad and inactive widget checks remain enforced, and the removed settings gap preference keeps all non-workflow route script budgets unchanged.

## Upstream baseline merge (`02e3ee0e5b44c90c1be4d959025449c56f27d6a7`)

`performance/architecture-baseline.json` is kept identical to `origin/main` (`02e3ee0e5b44c90c1be4d959025449c56f27d6a7`). In `performance/browser-baseline.json`, all upstream route timings, runner metadata, and non-workflow route baselines from `origin/main` are preserved unchanged (the +118 locale bytes in `en.json` stay well within the 4,096-byte route allowance); only the `editor-workflow` route (`empty` and `representative`) is updated for the lazy Workflow chunk delta (+6,140 script bytes, +118 locale bytes in `resourceBaseline.otherRawBytes`) and its two source owners:

| Activated metric (`editor-workflow`) | Upstream `origin/main` baseline | Updated with opt-in accordion |
| ------------------------------------ | ------------------------------: | ----------------------------: |
| Largest asset, raw bytes             |                         210,757 |                       216,897 |
| Script raw bytes                     |                         459,316 |                       465,456 |
| Total raw bytes                      |                         474,729 |                       480,869 |
| Requests                             |                               6 |                             6 |
| Script requests                      |                               5 |                             5 |

The only added source owners in `browser-baseline.json` are `src/features/workflow/ui/linear/formSections.ts` and `src/features/workflow/ui/linear/formSectionStatus.ts` inside the `editor-workflow-static` owner set. Other fixture owner sets and request counts are unchanged.

