# Prompt List Interaction Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give wildcard insertion rows, expanded-prompt preview rows, and prompt-template rows one shared hover, focus, radius, hit-area, and transition treatment without changing their height or content density.

**Architecture:** Reuse the existing Platform `Row` recipe as the single interaction primitive. Compose it through `asChild` around native buttons in the two surfaces still using Chakra ghost buttons; keep prompt templates as the reference implementation and retain all feature-specific content and state outside the shared primitive.

**Tech Stack:** React, TypeScript, Chakra UI v3, Vitest Browser Mode, Testing Library

## Global Constraints

- Preserve row height, padding, typography, thumbnails, numbering, wrapping, grouping, and click behavior.
- Keep wildcard and template edit/delete icon buttons as sibling controls.
- Keep the prompt-template `active="accent"` state and `aria-current`.
- Do not change the global ghost-button recipe or add new tokens, recipes, timing constants, or active states.
- Add browser-level regression coverage against browser-resolved styles, not duplicated production constants.

---

## Task 1: Lock down and implement the shared interaction contract

**Files:**

- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/WildcardsPanel.browser.test.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/DynamicPromptsButton.browser.test.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/WildcardsPanel.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/DynamicPromptsPanel.tsx`

- [ ] Add focused browser tests that identify the real selectable button in each list.
- [ ] Verify wildcard insertion, expanded-prompt application, and template application still call their existing handlers.
- [ ] Verify inactive rows resolve the same hover background, transition property/duration/timing, corner radius, and focus-visible outline as a shared `Row` probe or as each other.
- [ ] Verify the applied template remains `aria-current="true"` with its accent surface.
- [ ] Verify wildcard/template edit and delete controls remain outside the selectable button.
- [ ] Run the focused browser command and confirm the new consistency assertions fail for wildcard and expanded-prompt rows before production changes:

```bash
cd invokeai/frontend/webv2
pnpm run test:browser -- \
  src/features/generation/ui/promptFields/WildcardsPanel.browser.test.tsx \
  src/features/generation/ui/promptFields/DynamicPromptsButton.browser.test.tsx \
  src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
```
- [ ] Import the Platform `Row` primitive in both components.
- [ ] Replace only the wildcard insertion ghost `Button` with `Row asChild` wrapping a native `button type="button"`.
- [ ] Preserve the wildcard row's flex sizing, automatic height, padding, title, content, and insertion callback exactly.
- [ ] Leave wildcard edit/delete controls as siblings of the selectable row.
- [ ] Replace only the expanded-prompt ghost `Button` with `Row asChild` wrapping a native `button type="button"`.
- [ ] Preserve the preview row's automatic height, padding, title, index, highlighted prompt, and callback exactly.
- [ ] Remove the preview row's local `transitionDuration="faster"` override so the shared recipe owns motion.
- [ ] Do not modify `PromptTemplatesPanel.tsx` unless a test exposes a genuine contract mismatch.
- [ ] Run the focused browser command and confirm the new and existing tests pass.
- [ ] Run scoped static checks:

```bash
cd invokeai/frontend/webv2
pnpm exec oxfmt --check \
  src/features/generation/ui/promptFields/WildcardsPanel.tsx \
  src/features/generation/ui/promptFields/WildcardsPanel.browser.test.tsx \
  src/features/generation/ui/promptFields/DynamicPromptsPanel.tsx \
  src/features/generation/ui/promptFields/DynamicPromptsButton.browser.test.tsx \
  src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx
pnpm run lint:oxc
pnpm run lint:tsc
```

## Final verification

- [ ] Review the final diff against the approved design and confirm no row-density, content, account, persistence, or unrelated styling changes slipped in.
- [ ] Run repository verification:

```bash
cd invokeai/frontend/webv2
pnpm run lint
pnpm run test:all
git diff --check
```

- [ ] Record any pre-existing warnings or failures separately from regressions caused by this change.
- [ ] Commit the implementation with a focused message.
