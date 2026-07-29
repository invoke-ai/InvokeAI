# Prompt Template Image and Active-State Design

## Goal

Restore fetched prompt-template previews under React StrictMode and make the applied template visually unmistakable without adding one-off styling rules.

## Image lifecycle

`PromptTemplateImage` will continue fetching authenticated image blobs through TanStack Query. `BlobImage` will create its object URL inside its mount lifecycle, publish that URL to local state, and revoke that exact URL from the matching cleanup.

This makes StrictMode replay safe: the first setup URL may be revoked during the development replay, while the second setup creates a fresh live URL. Until a live URL exists, the component renders the caller-provided fallback. Final unmount revokes the live URL.

The regression test will render a fetched image inside `StrictMode` and prove:

- the URL used by the rendered image has not been revoked;
- replayed URLs are cleaned up;
- the live URL is revoked on final unmount;
- the existing semantic image outline remains applied.

## Active template row

The selectable portion of an applied template row will use the shared `Row` recipe with `active="accent"`, matching other active library rows. It will render as the existing button, expose `aria-current`, and use `accent.contrast` for both name and summary while active.

Edit and delete controls remain outside the accent surface so destructive and secondary actions keep their own interaction semantics. Inactive rows retain their current muted typography and hover behavior.

Browser coverage will apply a template and assert the selected button has the accent background, contrast text, and `aria-current`; another row must remain inactive.

## Boundaries

- No backend, query-key, authentication, or prompt-template DTO changes.
- No new color tokens or row recipes.
- Local preview data URLs keep their current behavior.
- The previously added `border.image` semantic outline remains the single image-outline rule.
