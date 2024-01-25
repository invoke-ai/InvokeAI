module.exports = {
  env: {
    browser: true,
    es6: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:react/jsx-runtime',
    'prettier',
    'plugin:storybook/recommended',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: 2018,
    sourceType: 'module',
  },
  plugins: [
    'react',
    '@typescript-eslint',
    'eslint-plugin-react-hooks',
    'i18next',
    'path',
    'unused-imports',
    'simple-import-sort',
    'eslint-plugin-import',
    // These rules are too strict for normal usage, but are useful for optimizing rerenders
    // '@arthurgeron/react-usememo',
  ],
  root: true,
  rules: {
    'path/no-relative-imports': ['error', { maxDepth: 0 }],
    curly: 'error',
    'i18next/no-literal-string': 'warn',
    'react/jsx-no-bind': ['error', { allowBind: true }],
    'react/jsx-curly-brace-presence': [
      'error',
      { props: 'never', children: 'never' },
    ],
    'react-hooks/exhaustive-deps': 'error',
    'no-var': 'error',
    'brace-style': 'error',
    'prefer-template': 'error',
    'import/no-duplicates': 'error',
    radix: 'error',
    'space-before-blocks': 'error',
    'import/prefer-default-export': 'off',
    '@typescript-eslint/no-unused-vars': 'off',
    'unused-imports/no-unused-imports': 'error',
    'unused-imports/no-unused-vars': [
      'warn',
      {
        vars: 'all',
        varsIgnorePattern: '^_',
        args: 'after-used',
        argsIgnorePattern: '^_',
      },
    ],
    // These rules are too strict for normal usage, but are useful for optimizing rerenders
    // '@arthurgeron/react-usememo/require-usememo': [
    //   'warn',
    //   {
    //     strict: false,
    //     checkHookReturnObject: false,
    //     fix: { addImports: true },
    //     checkHookCalls: false,

    //   },
    // ],
    // '@arthurgeron/react-usememo/require-memo': 'warn',
    '@typescript-eslint/ban-ts-comment': 'warn',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-empty-interface': [
      'error',
      {
        allowSingleExtends: true,
      },
    ],
    '@typescript-eslint/consistent-type-imports': [
      'error',
      {
        prefer: 'type-imports',
        fixStyle: 'separate-type-imports',
        disallowTypeAnnotations: true,
      },
    ],
    '@typescript-eslint/no-import-type-side-effects': 'error',
    'simple-import-sort/imports': 'error',
    'simple-import-sort/exports': 'error',
    // Prefer @invoke-ai/ui components over chakra
    'no-restricted-imports': 'off',
    '@typescript-eslint/no-restricted-imports': [
      'warn',
      {
        paths: [
          {
            name: '@chakra-ui/react',
            message: "Please import from '@invoke-ai/ui' instead.",
          },
          {
            name: '@chakra-ui/layout',
            message: "Please import from '@invoke-ai/ui' instead.",
          },
          {
            name: '@chakra-ui/portal',
            message: "Please import from '@invoke-ai/ui' instead.",
          },
        ],
      },
    ],
  },
  overrides: [
    {
      files: ['*.stories.tsx'],
      rules: {
        'i18next/no-literal-string': 'off',
      },
    },
    {
      files: ['**/*.ts', '**/*.tsx'],
      rules: {
        'react/prop-types': 'off',
      },
    },
  ],
  settings: {
    react: {
      version: 'detect',
    },
  },
};
