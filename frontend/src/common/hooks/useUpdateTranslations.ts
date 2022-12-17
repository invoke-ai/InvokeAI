import React from 'react';
import { useTranslation } from 'react-i18next';

export default function useUpdateTranslations(fn: () => void) {
  const { i18n } = useTranslation();
  const currentLang = localStorage.getItem('i18nextLng');

  React.useEffect(() => {
    fn();
  }, [fn]);

  React.useEffect(() => {
    i18n.on('languageChanged', () => {
      fn();
    });
  }, [fn, i18n, currentLang]);
}
