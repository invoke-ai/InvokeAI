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
  const language = useAppSelector((state) => state.system.language);
  const options = useMemo(
    () => [
      { label: t('common.langArabic'), value: 'ar' },
      { label: t('common.langDutch'), value: 'nl' },
      { label: t('common.langEnglish'), value: 'en' },
      { label: t('common.langFrench'), value: 'fr' },
      { label: t('common.langGerman'), value: 'de' },
      { label: t('common.langHebrew'), value: 'he' },
      { label: t('common.langItalian'), value: 'it' },
      { label: t('common.langJapanese'), value: 'ja' },
      { label: t('common.langKorean'), value: 'ko' },
      { label: t('common.langPolish'), value: 'pl' },
      { label: t('common.langBrPortuguese'), value: 'pt_BR' },
      { label: t('common.langPortuguese'), value: 'pt' },
      { label: t('common.langRussian'), value: 'ru' },
      { label: t('common.langSimplifiedChinese'), value: 'zh_CN' },
      { label: t('common.langSpanish'), value: 'es' },
      { label: t('common.langUkranian'), value: 'ua' },
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
