import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { languageChanged } from 'features/system/store/systemSlice';
import { isLanguage } from 'features/system/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const SettingsLanguageSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const language = useAppSelector((s) => s.system.language);
  const options = useMemo(
    () => [
      { label: 'العربية', value: 'ar' },
      { label: 'Azərbaycan dili', value: 'az' },
      { label: 'Deutsch', value: 'de' },
      { label: 'English', value: 'en' },
      { label: 'Español', value: 'es' },
      { label: 'Suomi', value: 'fi' },
      { label: 'Français', value: 'fr' },
      { label: 'עִבְֿרִית', value: 'he' },
      { label: 'Magyar Nyelv', value: 'hu' },
      { label: 'Italiano', value: 'it' },
      { label: '日本語', value: 'ja' },
      { label: '한국어', value: 'ko' },
      { label: 'Nederlands', value: 'nl' },
      { label: 'Polski', value: 'pl' },
      { label: 'Português', value: 'pt' },
      { label: 'Português do Brasil', value: 'pt_BR' },
      { label: 'Русский', value: 'ru' },
      { label: 'Svenska', value: 'sv' },
      { label: 'Türkçe', value: 'tr' },
      { label: 'Украї́нська', value: 'ua' },
      { label: '简体中文', value: 'zh_CN' },
      { label: '漢語', value: 'zh_Hant' },
    ],
    []
  );
  const isLocalizationEnabled = useFeatureStatus('localization').isFeatureEnabled;

  const value = useMemo(() => options.find((o) => o.value === language), [language, options]);

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
