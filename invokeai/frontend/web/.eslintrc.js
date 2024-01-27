module.exports = {
  extends: ['@invoke-ai/eslint-config-react'],
  rules: {
    // TODO(psyche): Enable this rule. Requires no default exports in components - many changes.
    'react-refresh/only-export-components': 'off',
    // TODO(psyche): Enable this rule. Requires a lot of eslint-disable-next-line comments.
    '@typescript-eslint/consistent-type-assertions': 'off',
  },
};
