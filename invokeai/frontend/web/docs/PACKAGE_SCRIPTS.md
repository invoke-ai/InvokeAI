# Package Scripts

WIP walkthrough of `package.json` scripts.

## `theme` & `theme:watch`

These run the Chakra CLI to generate types for the theme, or watch for code change and re-generate the types.

The CLI essentially monkeypatches Chakra's files in `node_modules`.

## `postinstall`

The `postinstall` script patches a few packages and runs the Chakra CLI to generate types for the theme.

### Patch `@chakra-ui/cli`

See: <https://github.com/chakra-ui/chakra-ui/issues/7394>
