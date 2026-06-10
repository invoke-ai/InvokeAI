# OXC Rule Compatibility

`webv2` uses `oxlint` and `oxfmt` instead of ESLint and Prettier.

The config mirrors the existing Invoke web lint intent where OXC supports equivalent rules:

- React and hooks checks: `react/jsx-no-bind`, `react/jsx-curly-brace-presence`, `react-hooks/*`.
- TypeScript checks: `typescript/consistent-type-imports`, `typescript/no-empty-interface`, plus TypeScript compile checks through `tsc --noEmit`.
- Import checks: `import/no-duplicates`, `import/no-cycle`.
- General correctness/style checks: `curly`, `no-var`, `prefer-template`, `radix`, `eqeqeq`, `no-eval`, `no-extend-native`, `no-implied-eval`, `no-label-var`, `no-return-assign`, `no-sequences`, `no-template-curly-in-string`, `no-throw-literal`, `no-unmodified-loop-condition`, `no-console`, `no-promise-executor-return`, and `require-await`.
- Formatting: `oxfmt` uses the same print width, tab width, semicolon, single quote, and trailing comma preferences as the existing web formatter. It uses `lf` line endings because `oxfmt` does not support Prettier's `auto` value.

Known unsupported or intentionally deferred equivalents:

- `brace-style`, `one-var`, and `react/jsx-no-bind`: oxlint 1.69.0 does not expose these ESLint/plugin rule names.
- `simple-import-sort/*`: oxlint does not currently provide the same configurable import sorting behavior.
- `unused-imports/no-unused-imports`: oxlint reports unused bindings, but it is not the same plugin rule.
- `@typescript-eslint/ban-ts-comment`: no exact oxlint equivalent is configured yet.
- `@typescript-eslint/no-import-type-side-effects`: no exact oxlint equivalent is configured yet.
- `@typescript-eslint/consistent-type-assertions`: no exact oxlint equivalent is configured yet.
- `path/no-relative-imports`: no exact oxlint equivalent is configured yet.
- `no-restricted-syntax`, `no-restricted-properties`, `no-restricted-imports`: no exact oxlint config equivalent is configured yet.
- `i18next/no-literal-string` and Storybook-specific overrides: not configured for the initial `webv2` shell.
