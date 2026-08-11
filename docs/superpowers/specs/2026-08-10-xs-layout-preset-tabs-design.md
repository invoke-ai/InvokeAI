# XS Layout Preset Tabs

## Context

The top-bar controls now use Chakra's `xs` size, which is 32px tall. Chakra's built-in Tabs recipe starts at `sm`, so the layout-preset tabs remain 36px tall and no longer align with their neighbours.

## Design

- Extend the shared Tabs slot recipe with a reusable `xs` size.
- Match the existing `Button size="xs"` metrics: a `sizes.8` (32px) control height, `xs` text, and compact `2.5` horizontal padding.
- Extend the local Tabs wrapper's `size` prop to accept `xs`, because Chakra's packaged Tabs types only list its built-in sizes.
- Change only the layout-preset strip from `size="sm"` to `size="xs"`. Existing Tabs consumers retain their current sizing.

## Verification

- Add a browser test that renders an `xs` tab beside an `xs` button and verifies both compute to 32px tall.
- Keep the existing Tabs interaction and accessibility coverage green.
- Run focused browser tests, TypeScript validation, formatting, and the relevant full verification suite.

## Non-goals

- Do not change Chakra's existing `sm`, `md`, or `lg` Tabs behavior.
- Do not modify other top-bar controls or opt other Tabs consumers into `xs`.
