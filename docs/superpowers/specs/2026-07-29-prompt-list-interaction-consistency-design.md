# Prompt List Interaction Consistency Design

## Goal

Make wildcard list items, expanded-prompt preview items, and prompt-template items feel like members of one interaction system while preserving the content density and typography appropriate to each list.

## Shared interaction contract

The selectable surface in all three lists will use the existing Platform `Row` recipe composed around a native button with `asChild`.

That gives every selectable item the same:

- `bg.muted` hover surface;
- two-pixel inset `accent.solid` keyboard-focus outline;
- `sm` corner radius;
- pointer cursor and full-width hit surface;
- background and color transition using `--wb-motion-duration-fast` (`120ms`, or `1ms` when reduced motion is enabled).

No new recipe, color token, duration token, or per-feature hover constant will be added.

The dynamic-prompts and prompt-template popovers use `bg.subtle` as their
backing surface. This keeps the shared `bg.muted` row hover visibly distinct
without overriding the `Row` recipe on individual list items.

## Component treatment

### Wildcard list

`WildcardRow` will replace its selectable ghost `Button` with `Row asChild` and a native `button`. The wildcard reference, values summary, padding, automatic height, and insert behavior remain unchanged.

Edit and delete icon buttons remain sibling controls outside the shared row surface. Their existing neutral and destructive button behavior is not changed.

### Expanded-prompt preview

`DynamicPromptRow` will replace its ghost `Button` with `Row asChild` and a native `button`. The numeric index, prompt highlighting, wrapping, padding, automatic height, and “use prompt” behavior remain unchanged.

The local `transitionDuration="faster"` override will be removed because the shared row recipe owns the transition.

### Prompt templates

`TemplateRow` already uses the shared `Row` recipe and remains the reference implementation. Applied templates keep `active="accent"`, `aria-current`, and contrast typography. Inactive templates use the same hover, focus, and transition treatment as the other two lists.

Thumbnail, text treatment, padding, automatic height, and edit/delete controls remain unchanged.

## Accessibility

Each selectable surface remains a native `button`, preserving keyboard activation and button semantics. The shared recipe supplies a consistent visible `:focus-visible` outline.

The template’s applied state remains exposed through `aria-current`. Wildcard and expanded-prompt items represent immediate actions rather than persistent selection, so they do not gain an active or current state.

## Testing

Browser coverage will render all three real list surfaces with the Chakra system and verify:

- wildcard, expanded-prompt, and inactive template buttons resolve to the same hover background;
- all three resolve the same transition property and duration;
- all three expose the shared focus-visible outline;
- wildcard insertion, expanded-prompt application, and template application still invoke their existing behaviors;
- an applied template retains its solid accent surface and `aria-current`;
- edit and delete controls remain outside the wildcard and template selectable surfaces.

Tests will compare browser-resolved styles against a shared `Row` probe or against each other rather than copying recipe implementation details into production code.

## Boundaries

- Do not change row height, padding, typography, thumbnails, numbering, text wrapping, or list grouping.
- Do not alter global Chakra ghost-button styling.
- Do not add active-state behavior to wildcard or expanded-prompt rows.
- Do not change prompt, wildcard, template, query, account, or persistence behavior.
- Do not introduce animation beyond the shared background/color transition.
