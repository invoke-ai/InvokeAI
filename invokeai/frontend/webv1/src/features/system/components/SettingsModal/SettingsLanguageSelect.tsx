import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { map } from 'es-toolkit/compat';
import { selectLanguage } from 'features/system/store/systemSelectors';
import { languageChanged } from 'features/system/store/systemSlice';
import type { Language } from 'features/system/store/types';
import { isLanguage } from 'features/system/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const optionsObject: Record<Language, string> = {
  ar: 'العربية',
  az: 'Azərbaycan dili',
  de: 'Deutsch',
  en: 'English',
  es: 'Español',
  fi: 'Suomi',
  fr: 'Français',
  he: 'עִבְֿרִית',
  hu: 'Magyar Nyelv',
  it: 'Italiano',
  ja: '日本語',
  ko: '한국어',
  nl: 'Nederlands',
  pl: 'Polski',
  pt: 'Português',
  'pt-BR': 'Português do Brasil',
  ru: 'Русский',
  sv: 'Svenska',
  tr: 'Türkçe',
  ua: 'Украї́нська',
  vi: 'Tiếng Việt',
  'zh-CN': '简体中文',
  'zh-Hant': '漢語',
};

const options = map(optionsObject, (label, value) => ({ label, value }));

export const SettingsLanguageSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const language = useAppSelector(selectLanguage);

  const value = useMemo(() => options.find((o) => o.value === language), [language]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isLanguage(v?.value)) {
        return;
      }
      dispatch(languageChanged(v.value));
    },
    [dispatch]
  );
  return (
    <FormControl>
      <FormLabel>{t('common.languagePickerLabel')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

SettingsLanguageSelect.displayName = 'SettingsLanguageSelect';
