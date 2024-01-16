import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOnChange } from 'common/components/InvSelect/types';
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
      { label: t('common.langArabic', { lng: 'ar' }), value: 'ar' },
      { label: t('common.langDutch', { lng: 'nl' }), value: 'nl' },
      { label: t('common.langEnglish', { lng: 'en' }), value: 'en' },
      { label: t('common.langFrench', { lng: 'fr' }), value: 'fr' },
      { label: t('common.langGerman', { lng: 'de' }), value: 'de' },
      { label: t('common.langHebrew', { lng: 'he' }), value: 'he' },
      { label: t('common.langItalian', { lng: 'it' }), value: 'it' },
      { label: t('common.langJapanese', { lng: 'ja' }), value: 'ja' },
      { label: t('common.langKorean', { lng: 'ko' }), value: 'ko' },
      { label: t('common.langPolish', { lng: 'pl' }), value: 'pl' },
      { label: t('common.langBrPortuguese', { lng: 'pt_BR' }), value: 'pt_BR' },
      { label: t('common.langPortuguese', { lng: 'pt' }), value: 'pt' },
      { label: t('common.langRussian', { lng: 'ru' }), value: 'ru' },
      {
        label: t('common.langSimplifiedChinese', { lng: 'zh_CN' }),
        value: 'zh_CN',
      },
      { label: t('common.langSpanish', { lng: 'es' }), value: 'es' },
      { label: t('common.langUkranian', { lng: 'ua' }), value: 'ua' },
    ],
    [t]
  );
  const isLocalizationEnabled =
    useFeatureStatus('localization').isFeatureEnabled;

  const value = useMemo(
    () => options.find((o) => o.value === language),
    [language, options]
  );

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isLanguage(v?.value)) {
        return;
      }
      dispatch(languageChanged(v.value));
    },
    [dispatch]
  );
  return (
    <InvControl
      label={t('common.languagePickerLabel')}
      isDisabled={!isLocalizationEnabled}
    >
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
});

SettingsLanguageSelect.displayName = 'SettingsLanguageSelect';
