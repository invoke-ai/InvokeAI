import i18n from '@platform/i18n/client';
import { getWorkbenchLanguageDirection } from '@platform/i18n/languages';
import { useWorkbenchPreferenceSelector, useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useLayoutEffect } from 'react';

const LANGUAGE_HINT_STORAGE_KEY = 'invokeai:v7:webv2:language';

export const I18nController = () => {
  const language = useWorkbenchPreferenceSelector((preferences) => preferences.language);
  const status = useWorkbenchSettingsSelector((snapshot) => snapshot.status);
  const hasResolved = status === 'ready' || status === 'error';

  useLayoutEffect(() => {
    if (!hasResolved) {
      return;
    }

    const direction = getWorkbenchLanguageDirection(language);
    const root = document.documentElement;

    root.lang = language;
    root.dir = direction;
    document.body.dir = direction;
    void i18n.changeLanguage(language);

    try {
      window.localStorage.setItem(LANGUAGE_HINT_STORAGE_KEY, language);
    } catch {
      // Storage unavailable; the next load paints with the default language.
    }
  }, [hasResolved, language]);

  return null;
};
