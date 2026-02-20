import js from '@eslint/js';
import typescriptEslint from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import pluginI18Next from 'eslint-plugin-i18next';
import pluginImport from 'eslint-plugin-import';
import pluginPath from 'eslint-plugin-path';
import pluginReact from 'eslint-plugin-react';
import pluginReactHooks from 'eslint-plugin-react-hooks';
import pluginReactRefresh from 'eslint-plugin-react-refresh';
import pluginSimpleImportSort from 'eslint-plugin-simple-import-sort';
import pluginStorybook from 'eslint-plugin-storybook';
import pluginUnusedImports from 'eslint-plugin-unused-imports';
import globals from 'globals';

export default [
  js.configs.recommended,

  {
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        ...globals.browser,
        ...globals.node,
        GlobalCompositeOperation: 'readonly',
        RequestInit: 'readonly',
      },
    },

    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],

    plugins: {
      react: pluginReact,
      '@typescript-eslint': typescriptEslint,
      'react-hooks': pluginReactHooks,
      import: pluginImport,
      'unused-imports': pluginUnusedImports,
      'simple-import-sort': pluginSimpleImportSort,
      'react-refresh': pluginReactRefresh.configs.vite,
      path: pluginPath,
      i18next: pluginI18Next,
      storybook: pluginStorybook,
    },

    rules: {
      ...typescriptEslint.configs.recommended.rules,
      ...pluginReact.configs.recommended.rules,
      ...pluginReact.configs['jsx-runtime'].rules,
      ...pluginReactHooks.configs.recommended.rules,
      ...pluginStorybook.configs.recommended.rules,

      'react/jsx-no-bind': [
        'error',
        {
          allowBind: true,
          allowArrowFunctions: true,
        },
      ],

      'react/jsx-curly-brace-presence': [
        'error',
        {
          props: 'never',
          children: 'never',
        },
      ],

      'react-hooks/exhaustive-deps': 'error',

      curly: 'error',
      'no-var': 'error',
      'brace-style': 'error',
      'prefer-template': 'error',
      radix: 'error',
      'space-before-blocks': 'error',
      eqeqeq: 'error',
      'one-var': ['error', 'never'],
      'no-eval': 'error',
      'no-extend-native': 'error',
      'no-implied-eval': 'error',
      'no-label-var': 'error',
      'no-return-assign': 'error',
      'no-sequences': 'error',
      'no-template-curly-in-string': 'error',
      'no-throw-literal': 'error',
      'no-unmodified-loop-condition': 'error',
      'import/no-duplicates': 'error',
      'import/prefer-default-export': 'off',
      'unused-imports/no-unused-imports': 'error',

      'unused-imports/no-unused-vars': [
        'error',
        {
          vars: 'all',
          varsIgnorePattern: '^_',
          args: 'after-used',
          argsIgnorePattern: '^_',
        },
      ],

      'simple-import-sort/imports': 'error',
      'simple-import-sort/exports': 'error',
      '@typescript-eslint/no-unused-vars': 'off',

      '@typescript-eslint/ban-ts-comment': [
        'error',
        {
          'ts-expect-error': 'allow-with-description',
          'ts-ignore': true,
          'ts-nocheck': true,
          'ts-check': false,
          minimumDescriptionLength: 10,
        },
      ],

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

      '@typescript-eslint/consistent-type-assertions': [
        'error',
        {
          assertionStyle: 'as',
        },
      ],

      'path/no-relative-imports': [
        'error',
        {
          maxDepth: 0,
        },
      ],

      'no-console': 'warn',
      'no-promise-executor-return': 'error',
      'require-await': 'error',

      'no-restricted-syntax': [
        'error',
        {
          selector: 'CallExpression[callee.name="setActiveTab"]',
          message:
            'setActiveTab() can only be called from use-navigation-api.tsx. Use navigationApi.switchToTab() instead.',
        },
      ],

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

      // Typescript handles this for us: https://eslint.org/docs/latest/rules/no-redeclare#handled_by_typescript
      'no-redeclare': 'off',

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
            {
              name: 'zod/v3',
              message: 'Import from zod instead.',
            },
          ],
        },
      ],
    },

    settings: {
      react: {
        version: 'detect',
      },
    },
  },

  {
    files: ['**/use-navigation-api.tsx'],
    rules: {
      'no-restricted-syntax': 'off',
    },
  },

  {
    files: ['**/*.stories.tsx'],
    rules: {
      'i18next/no-literal-string': 'off',
    },
  },

  {
    ignores: [
      '**/dist/',
      '**/static/',
      '**/.husky/',
      '**/node_modules/',
      '**/patches/',
      '**/stats.html',
      '**/index.html',
      '**/.yarn/',
      '**/*.scss',
      'src/services/api/schema.ts',
      '.prettierrc.js',
      '.storybook',
    ],
  },
];
