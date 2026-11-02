# Workflow accordion resource measurements

The accordion adds code for heading-based sections, saved expansion state, hidden-field validation and layout-aware boundaries. The activated Workflow widget uses this code; the Launchpad and inactive widget checks remain enforced.

## Reproduction

The failing GitHub run was reproduced before the baseline refresh. On the original branch, both workflow fixtures loaded an activated largest asset of 214,578 bytes, 460,123 script bytes and 475,536 total bytes, matching the CI failure. An unchanged checkout of the original PR base reproduced a separate stale Canvas-baseline difference; that difference was not treated as an accordion defect.

The branch was then updated to upstream `35d08c870cdc79ffbef30b64d1eb9331d86b0858` without rewriting its published commits. Its frontend sources match the official alpha.4 release base. The feature was measured again against the newer baseline.

## Current measurement

Both `editor-workflow/empty` and `editor-workflow/representative` produced:

| Activated metric         | Recorded upstream baseline | Unchanged alpha.4 control | Measured with accordion |
| ------------------------ | -------------------------: | ------------------------: | ----------------------: |
| Largest asset, raw bytes |                    209,892 |                   209,818 |                 215,532 |
| Script raw bytes         |                    454,931 |                   454,789 |                 460,503 |
| Total raw bytes          |                    470,344 |                   470,202 |                 475,916 |
| Requests                 |                          6 |                         6 |                       6 |
| Script requests          |                          5 |                         5 |                       5 |

The unchanged alpha.4 control and feature were built with the same installed dependencies. The incremental activated Workflow payload is 5,714 raw bytes, without extra requests. The settings preference and English strings also contribute 808 shared raw bytes in the browser journey (597 script bytes and 211 other bytes).

The only added source owners are `src/features/workflow/ui/linear/formSections.ts` and `src/features/workflow/ui/linear/formSectionStatus.ts`, in the Workflow fixture's owner set. Other fixture owner sets are unchanged.

## Recording and verification

The browser baseline was recorded using the documented `pnpm run test:performance:browser:update-baseline` command: one warmup, one trace sample and five scored samples per fixture. A subsequent normal two-observation browser gate passed all ten fixtures.

This is a complete baseline recording: the diff includes measurements for other routes and static resources, as well as Workflow activation and the two added source owners. It is not limited to the Workflow byte values.

The generated measurements are from Chromium 1248 on Windows. Timing values are environment-specific observations, not a claim of a speed improvement; the existing timing policy remains unchanged. The budget derivation, hard ceilings, request-count comparison rules, readiness marks and inactive-widget checks are unchanged.

The independent unchanged-alpha.4 reference comparison and full source-scanner verification passed outside the development sandbox, whose path policy blocks traversal of the public source folder named `credentials`. Both performance gates passed with the unchanged control reference applied, retaining the lower request-count limits.

The same verification also passed 146 targeted unit/architecture tests, 44 browser tests, the project-file journey and 21 accessibility journeys. A separate full unit run passed 12,415 of 12,420 tests. Its five Windows/locale-related failures were reproduced in the unmodified alpha.4 control: AltGr matching and numeric formatting in four existing test files. Those unrelated application files were not changed; a green full release suite is not claimed.
