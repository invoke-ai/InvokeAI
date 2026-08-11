# XS Layout Preset Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable 32px `xs` Tabs size and use it for the layout-preset tabs so they align with the other top-bar controls.

**Architecture:** Extend the existing shared Chakra Tabs slot recipe instead of applying local dimensions. Narrowly widen the local Tabs wrapper's size type for the custom variant, then opt only the layout-preset strip into it.

**Tech Stack:** React 19, TypeScript, Chakra UI 3 slot recipes, Vitest Browser Mode, pnpm

## Global Constraints

- `Tabs size="xs"` must compute to the same 32px height as `Button size="xs"`.
- Existing `sm`, `md`, and `lg` Tabs consumers must remain unchanged.
- Only the layout-preset strip opts into `xs` in this change.
- Preserve all concurrent top-bar edits.

---

### Task 1: Add and consume the shared `xs` Tabs size

**Files:**
- Modify: `invokeai/frontend/webv2/src/platform/ui/Tabs.browser.test.tsx`
- Modify: `invokeai/frontend/webv2/src/platform/ui/theme/recipes.ts`
- Modify: `invokeai/frontend/webv2/src/platform/ui/Tabs.tsx`
- Modify: `invokeai/frontend/webv2/src/workbench/shell/topbar/LayoutPresetStrip.tsx`

**Interfaces:**
- Consumes: Chakra's existing Tabs slot recipe and its `sizes.8`, `spacing.2.5`, and `xs` text-style tokens.
- Produces: `<Tabs.Root size="xs">`, a typed project-level Tabs API whose trigger height is 32px.

- [ ] **Step 1: Write the failing browser test**

Add a size test that renders the project `Button` and `Tabs` wrappers together:

```tsx
<Stack align="start">
  <Button aria-label="xs button" size="xs">
    Action
  </Button>
  <Tabs.Root size="xs" value="preset">
    <Tabs.List>
      <Tabs.Trigger aria-label="xs tab" value="preset">
        Preset
      </Tabs.Trigger>
    </Tabs.List>
  </Tabs.Root>
</Stack>
```

Assert that both controls have a computed height of `32px`, that their heights match, and that the tab uses `10px` inline padding.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
pnpm exec vitest run --config vitest.browser.config.mts src/platform/ui/Tabs.browser.test.tsx
```

Expected: FAIL because Chakra has no `xs` Tabs recipe, so the trigger does not compute to the required 32px metrics.

- [ ] **Step 3: Implement the minimal shared size**

Add this variant beside the existing Tabs sizes in `theme/recipes.ts`:

```ts
xs: {
  root: {
    '--tabs-height': 'sizes.8',
    '--tabs-content-padding': 'spacing.2.5',
  },
  trigger: { px: '2.5', py: '0.5', textStyle: 'xs' },
},
```

In `Tabs.tsx`, preserve Chakra's existing root props while adding `xs` to the local wrapper contract:

```ts
type ChakraTabsRootProps = ComponentProps<typeof ChakraTabs.Root>;
type TabsRootProps = Omit<ChakraTabsRootProps, 'size'> & {
  size?: ChakraTabsRootProps['size'] | 'xs';
};

const Root = ({ size, ...props }: TabsRootProps) => (
  <ChakraTabs.Root colorPalette="accent" size={size as ChakraTabsRootProps['size']} {...props} />
);
```

Change `LayoutPresetStrip.tsx` from `size="sm"` to `size="xs"` on its `Tabs.Root`. Do not alter the surrounding top-bar sizing edits.

- [ ] **Step 4: Run focused verification to verify GREEN**

Run:

```bash
pnpm exec vitest run --config vitest.browser.config.mts src/platform/ui/Tabs.browser.test.tsx
pnpm run lint:tsc
pnpm run format:check
```

Expected: the Tabs browser file passes, TypeScript accepts `size="xs"`, and formatting is clean.

- [ ] **Step 5: Run regression verification**

Run:

```bash
pnpm run lint
pnpm run test:all
```

Expected: all static, unit, browser, architecture, and fixture checks pass.

- [ ] **Step 6: Commit only this task's files**

```bash
git add docs/superpowers/plans/2026-08-10-xs-layout-preset-tabs.md invokeai/frontend/webv2/src/platform/ui/Tabs.browser.test.tsx invokeai/frontend/webv2/src/platform/ui/theme/recipes.ts invokeai/frontend/webv2/src/platform/ui/Tabs.tsx invokeai/frontend/webv2/src/workbench/shell/topbar/LayoutPresetStrip.tsx
git commit -m "feat(ui): add xs tabs size"
```
