# Global Row Hover Contrast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every platform `Row` a clearly visible hover state while returning prompt popovers to the standard surface color.

**Architecture:** The shared recipe owns global row interaction styling. Prompt popovers use the same surface token as other popovers, and browser tests resolve semantic tokens through Chakra rather than asserting raw colors.

**Tech Stack:** React, TypeScript, Chakra UI 3.36, Vitest Browser

## Global Constraints

- Use `bg.emphasized` for inactive `Row` hover while preserving active-state fills and focus treatment.
- Use `bg.muted` for both prompt popovers; add no local override, Row variant, or theme token.

---

### Task 1: Lock down the intended surface and hover colors

**Files:**
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/DynamicPromptsButton.browser.test.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplatesButton.browser.test.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplates.browser.test.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/WildcardsPanel.browser.test.tsx`

**Interfaces:**
- Consumes: Chakra semantic color tokens and existing row-interaction helpers.
- Produces: Browser regressions for `bg.emphasized` row hover and `bg.muted` popover content.

- [x] Replace each row-hover probe with `<Box bg="bg.emphasized" data-testid="row-hover-style-probe" />`.
- [x] Add a `bg.muted` surface probe to each popover-button harness and compare it with `[data-scope="popover"][data-part="content"]` after opening.
- [x] Run the focused browser command and confirm failures report `bg.muted` row hover and `bg.subtle` popovers.

### Task 2: Correct the shared interaction and surface contracts

**Files:**
- Modify: `invokeai/frontend/webv2/src/platform/ui/theme/recipes.ts`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/DynamicPromptsButton.tsx`
- Modify: `invokeai/frontend/webv2/src/features/generation/ui/promptFields/PromptTemplatesButton.tsx`

**Interfaces:**
- Consumes: Existing `rowRecipe`, `bg.emphasized`, and `bg.muted`.
- Produces: A global emphasized hover and standard prompt popover surfaces.

- [x] Change `rowRecipe.base._hover` to `{ bg: 'bg.emphasized' }`.
- [x] Change both `Popover.Content` backgrounds to `bg="bg.muted"`.
- [x] Run the focused browser command from Task 1 and confirm all tests pass.
- [x] Run `pnpm run check:architecture`, `git diff --check`, and the merge-base line-count audit.
