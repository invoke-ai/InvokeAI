# Package Scripts

WIP walkthrough of `package.json` scripts.

## `theme` & `theme:watch`

These run the Chakra CLI to generate types for the theme, or watch for code change and re-generate the types.

The CLI essentially monkeypatches Chakra's files in `node_modules`.

## `postinstall`

The `postinstall` script patches a few packages and runs the Chakra CLI to generate types for the theme.

### Patch `@chakra-ui/cli`

See: <https://github.com/chakra-ui/chakra-ui/issues/7394>

### Patch `redux-persist`

We want to persist the canvas state to `localStorage` but many canvas operations change data very quickly, so we need to debounce the writes to `localStorage`.

`redux-persist` is unfortunately unmaintained. The repo's current code is nonfunctional, but the last release's code depends on a package that was removed from `npm` for being malware, so we cannot just fork it.

So, we have to patch it directly. Perhaps a better way would be to write a debounced storage adapter, but I couldn't figure out how to do that.

### Patch `redux-deep-persist`

This package makes blacklisting and whitelisting persist configs very simple, but we have to patch it to match `redux-persist` for the types to work.
