module.exports = {
  trailingComma: 'es5',
  tabWidth: 2,
  endOfLine: 'crlf',
  semi: true,
  singleQuote: true,
  overrides: [
    {
      files: ['public/locales/*.json'],
      options: {
        tabWidth: 4,
      },
    },
  ],
};
