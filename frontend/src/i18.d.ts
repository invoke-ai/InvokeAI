import 'i18next';

// Import All Namespaces (for the default language only)
import common from '../public/locales/common/en.json';
import unifiedcanvas from '../public/locales/unifiedcanvas/en.json';
import options from '../public/locales/options/en.json';
import gallery from '../public/locales/gallery/en.json';
import toast from '../public/locales/toast/en.json';

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
      gallery: typeof gallery;
      toast: typeof toast;
    };
    // Never Return Null
    returnNull: false;
  }
}
