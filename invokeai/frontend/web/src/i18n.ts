import i18n from 'i18next';
import Backend from 'i18next-http-backend';
import { initReactI18next } from 'react-i18next';

// TODO: Disabled for IDE performance issues with our translation JSON
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import translationEN from '../public/locales/en.json';

if (import.meta.env.MODE === 'package') {
  i18n.use(initReactI18next).init({
    lng: 'en',
    resources: {
      en: { translation: translationEN },
    },
    debug: false,
    interpolation: {
      escapeValue: false,
    },
    returnNull: false,
  });
} else {
  i18n
    .use(Backend)
    // .use(
    //   new LanguageDetector(null, {
    //     lookupLocalStorage: `${LOCALSTORAGE_PREFIX}lng`,
    //   })
    // )
    .use(initReactI18next)
    .init({
      fallbackLng: 'en',
      debug: false,
      backend: {
        loadPath: '/locales/{{lng}}.json',
      },
      interpolation: {
        escapeValue: false,
      },
      returnNull: false,
    });
}

export default i18n;
