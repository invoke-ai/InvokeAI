module.exports = {
  trailingComma: 'es5',
  tabWidth: 2,
  semi: true,
  singleQuote: true,
  endOfLine: 'auto',
  overrides: [
    {
      files: ['public/locales/*.json'],
      options: {
        tabWidth: 4,
      },
    },
    {
      files: ['**/*.md'],
      options: {
        tabWidth: 2,
        printWidth: 100,
      },
    },
  ],
};
