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
    // TODO: ENABLE THIS RULE BEFORE v6.0.0
    // 'i18next/no-literal-string': 'error',
    // https://eslint.org/docs/latest/rules/no-console
    'no-console': 'warn',
    // https://eslint.org/docs/latest/rules/no-promise-executor-return
    'no-promise-executor-return': 'error',
    // https://eslint.org/docs/latest/rules/require-await
    'require-await': 'error',
    // TODO: ENABLE THIS RULE BEFORE v6.0.0
    'react/display-name': 'off',
    'no-restricted-properties': [
      'error',
      {
        object: 'crypto',
        property: 'randomUUID',
        message: 'Use of crypto.randomUUID is not allowed as it is not available in all browsers.',
      },
      {
        object: 'navigator',
        property: 'clipboard',
        message:
          'The Clipboard API is not available by default in Firefox. Use the `useClipboard` hook instead, which wraps clipboard access to prevent errors.',
      },
    ],
    'no-restricted-imports': [
      'error',
      {
        paths: [
          {
            name: 'lodash-es',
            importNames: ['isEqual'],
            message: 'Please use objectEquals from @observ33r/object-equals instead.',
          },
          {
            name: 'lodash-es',
            message: 'Please use es-toolkit instead.',
          },
          {
            name: 'es-toolkit',
            importNames: ['isEqual'],
            message: 'Please use objectEquals from @observ33r/object-equals instead.',
          },
        ],
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
