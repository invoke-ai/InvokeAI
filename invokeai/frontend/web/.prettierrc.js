module.exports = {
  ...require('@invoke-ai/prettier-config-react'),
  overrides: [
    {
      files: ['public/locales/*.json'],
      options: {
        tabWidth: 4,
      },
    },
  ],
};
