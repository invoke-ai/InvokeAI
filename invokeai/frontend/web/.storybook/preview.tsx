import type { Preview } from '@storybook/react-vite';
import { themes } from 'storybook/theming';
import { $store } from 'app/store/nanostores/store';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { Provider } from 'react-redux';

// TODO: Disabled for IDE performance issues with our translation JSON
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import translationEN from '../public/locales/en.json';
import ThemeLocaleProvider from '../src/app/components/ThemeLocaleProvider';
import { $baseUrl } from '../src/app/store/nanostores/baseUrl';
import { createStore } from '../src/app/store/store';
import { ReduxInit } from './ReduxInit';

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
$store.set(store);
$baseUrl.set('http://localhost:9090');

const preview: Preview = {
  decorators: [
    (Story) => {
      return (
        <Provider store={store}>
          <ThemeLocaleProvider>
            <ReduxInit>
              <Story />
            </ReduxInit>
          </ThemeLocaleProvider>
        </Provider>
      );
    },
  ],
  parameters: {
    docs: {
      theme: themes.dark,
      codePanel: true
    },
  },
};

export default preview;
