import 'i18next';

declare module 'i18next' {
  // Extend CustomTypeOptions
  interface CustomTypeOptions {
    // Setting Default Namespace As English
    defaultNS: 'en';
    // Custom Types For Resources
    // Never Return Null
    returnNull: false;
  }
}
