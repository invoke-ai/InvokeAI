module.exports = {
  extends: ['@invoke-ai/eslint-config-react'],
  plugins: ['path', 'i18next'],
  rules: {
    // TODO(psyche): Enable this rule. Requires no default exports in components - many changes.
    'react-refresh/only-export-components': 'off',
    // TODO(psyche): Enable this rule. Requires a lot of eslint-disable-next-line comments.
    '@typescript-eslint/consistent-type-assertions': 'off',
    // https://github.com/qdanik/eslint-plugin-path
    'path/no-relative-imports': ['error', { maxDepth: 0 }],
    // https://github.com/edvardchen/eslint-plugin-i18next/blob/HEAD/docs/rules/no-literal-string.md
    'i18next/no-literal-string': 'error',
    // https://eslint.org/docs/latest/rules/no-console
    'no-console': 'error',
    // https://eslint.org/docs/latest/rules/no-promise-executor-return
    'no-promise-executor-return': 'error',
    // https://eslint.org/docs/latest/rules/require-await
    'require-await': 'error',
    'no-restricted-properties': [
      'error',
      {
        object: 'crypto',
        property: 'randomUUID',
        message: 'Use of crypto.randomUUID is not allowed as it is not available in all browsers.',
      },
    ],
  },
  overrides: [
    /**
     * Overrides for stories
     */
    {
      files: ['*.stories.tsx'],
      rules: {
        // We may not have i18n available in stories.
        'i18next/no-literal-string': 'off',
      },
    },
  ],
};
