import { Preview } from '@storybook/react';
import { themes } from '@storybook/theming';
import i18n from 'i18next';
import React from 'react';
import { initReactI18next } from 'react-i18next';
import { Provider } from 'react-redux';
import GlobalHotkeys from '../src/app/components/GlobalHotkeys';
import ThemeLocaleProvider from '../src/app/components/ThemeLocaleProvider';
import { createStore } from '../src/app/store/store';
// TODO: Disabled for IDE performance issues with our translation JSON
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import translationEN from '../public/locales/en.json';

i18n.use(initReactI18next).init({
  lng: 'en',
  resources: {
    en: { translation: translationEN },
  },
  debug: true,
  interpolation: {
    escapeValue: false,
  },
  returnNull: false,
});

const store = createStore(undefined, false);

const preview: Preview = {
  decorators: [
    (Story) => (
      <Provider store={store}>
        <ThemeLocaleProvider>
          <GlobalHotkeys />
          <Story />
        </ThemeLocaleProvider>
      </Provider>
    ),
  ],
  parameters: {
    docs: {
      theme: themes.dark,
    },
  },
};

export default preview;
