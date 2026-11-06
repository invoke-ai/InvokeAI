import i18n from '@platform/i18n/client';
import { getEffectiveWorkbenchLanguage, getWorkbenchLanguageDirection } from '@platform/i18n/languages';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getWorkbenchSettingsSnapshot, subscribeWorkbenchSettings } from '@workbench/settings/store';

const LANGUAGE_HINT_STORAGE_KEY = 'invokeai:v7:webv2:language';

export const I18nController = () => {
  useMountEffect(() => {
    let active = true;
    let running = false;
    let requested: string | undefined;
    let savedPreference: string | undefined;
    const apply = async () => {
      if (running) {
        return;
      }
      running = true;
      try {
        while (true) {
          if (!active || !requested || i18n.language === requested) {
            break;
          }
          const next = requested;
          await i18n.changeLanguage(next);
        }
        if (!active || !requested) {
          return;
        }
        const direction = getWorkbenchLanguageDirection(
          getEffectiveWorkbenchLanguage(getWorkbenchSettingsSnapshot().preferences.language)
        );
        document.documentElement.lang = requested;
        document.documentElement.dir = direction;
        document.body.dir = direction;
      } finally {
        running = false;
      }
    };
    const sync = () => {
      const { status, preferences } = getWorkbenchSettingsSnapshot();
      if (status !== 'ready' && status !== 'error') {
        return;
      }
      if (savedPreference === preferences.language) {
        return;
      }
      savedPreference = preferences.language;
      requested = getEffectiveWorkbenchLanguage(preferences.language);
      try {
        localStorage.setItem(LANGUAGE_HINT_STORAGE_KEY, preferences.language);
      } catch {
        /* Storage is optional. */
      }
      void apply().catch(() => undefined);
    };
    const unsubscribe = subscribeWorkbenchSettings(sync);
    sync();
    return () => {
      active = false;
      unsubscribe();
    };
  });
  return null;
};
