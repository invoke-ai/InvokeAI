module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint', 'eslint-plugin-react-hooks'],
  root: true,
  rules: {
    '@typescript-eslint/no-unused-vars': ['warn', { varsIgnorePattern: '_+' }],
  },
};
