import 'i18next';

// Import All Namespaces (for the default language only)
import common from './locales/common/en.json';
import unifiedcanvas from './locales/unifiedcanvas/en.json';
import options from './locales/options/en.json';

declare module 'i18next' {
  // Extend CustomTypeOptions
  interface CustomTypeOptions {
    // Setting Default Namespace As English
    defaultNS: 'en';
    // Custom Types For Resources
    resources: {
      common: typeof common;
      unifiedcanvas: typeof unifiedcanvas;
      options: typeof options;
    };
    // Never Return Null
    returnNull: false;
  }
}
