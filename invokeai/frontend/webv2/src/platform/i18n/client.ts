import { getDeploymentBaseUrl } from '@platform/transport/deploymentBase';
import i18n from 'i18next';
import Backend from 'i18next-http-backend';
import { initReactI18next } from 'react-i18next';

void i18n
  .use(Backend)
  .use(initReactI18next)
  .init({
    backend: {
      loadPath: (_languages: readonly string[], namespaces: readonly string[]) =>
        `${getDeploymentBaseUrl()}/locales/{{lng}}${namespaces[0] === 'fonts' ? '.fonts' : ''}.json`,
      // Changes with locale content, so a release never reads the previous one's cached translations; the service
      // worker caches only these versioned URLs.
      queryStringParams: { v: import.meta.env.LOCALE_VERSION },
    },
    debug: false,
    fallbackLng: 'en',
    fallbackNS: 'translation',
    ns: ['translation'],
    defaultNS: 'translation',
    interpolation: {
      escapeValue: false,
    },
    returnNull: false,
  });

export default i18n;
