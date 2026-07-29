# Global Row Hover Contrast

## Goal

Make interactive `Row` hover feedback clearly visible without changing the
background of the surface containing the row.

## Design

The shared `rowRecipe` will use `bg.emphasized` for the inactive hover state.
This matches the existing `dropdownItem` treatment and keeps hover contrast
consistent across light, dark, and classic themes.

The `brand` and `accent` active variants keep their existing variant-specific
hover fills. Focus continues to use the inset `accent.solid` outline, so hover
and keyboard focus remain independently visible.

Prompt-template and dynamic-prompt popovers will return from `bg.subtle` to the
standard `bg.muted` popover surface. The wildcard panel inherits the
dynamic-prompt popover surface.

No new Row variant, feature-local hover override, or theme token is introduced.

## Scope

The stronger hover applies to every consumer of the platform `Row`, including
prompt lists, model and node libraries, queue rows, project rows, widget bars,
and layer rows.

## Verification

Browser tests will assert that prompt-list rows resolve `bg.emphasized` on
hover while their popover content resolves `bg.muted`. Existing focus, active,
pointer, and reduced-motion assertions remain intact.

Focused browser tests run first, followed by the frontend architecture gate and
the PR line-budget audit.
