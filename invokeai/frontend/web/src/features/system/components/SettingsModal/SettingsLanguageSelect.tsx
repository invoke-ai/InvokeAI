import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { languageChanged } from 'features/system/store/systemSlice';
import type { Language } from 'features/system/store/types';
import { isLanguage } from 'features/system/store/types';
import { map } from 'lodash-es';
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
  pt_BR: 'Português do Brasil',
  ru: 'Русский',
  sv: 'Svenska',
  tr: 'Türkçe',
  ua: 'Украї́нська',
  zh_CN: '简体中文',
  zh_Hant: '漢語',
};

const options = map(optionsObject, (label, value) => ({ label, value }));

export const SettingsLanguageSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const language = useAppSelector((s) => s.system.language);
  const isLocalizationEnabled = useFeatureStatus('localization').isFeatureEnabled;

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
    <FormControl isDisabled={!isLocalizationEnabled}>
      <FormLabel>{t('common.languagePickerLabel')}</FormLabel>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

SettingsLanguageSelect.displayName = 'SettingsLanguageSelect';
