import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import Backend from 'i18next-http-backend';
import { initReactI18next } from 'react-i18next';

i18n
  .use(Backend)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    fallbackLng: 'en',
    debug: true,
    backend: {
      loadPath: 'src/locales/{{ns}}/{{lng}}.json',
    },
    ns: ['common', 'options', 'unifiedcanvas', 'gallery'],
    returnNull: false,
  });

export default i18n;
