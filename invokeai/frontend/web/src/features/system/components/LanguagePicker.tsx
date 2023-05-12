import type { ReactNode } from 'react';

import { VStack } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { useTranslation } from 'react-i18next';
import { FaCheck, FaLanguage } from 'react-icons/fa';
import i18n from 'i18n';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { languageSelector } from '../store/systemSelectors';
import { languageChanged } from '../store/systemSlice';

export const LANGUAGES = {
  ar: i18n.t('common.langArabic', { lng: 'ar' }),
  nl: i18n.t('common.langDutch', { lng: 'nl' }),
  en: i18n.t('common.langEnglish', { lng: 'en' }),
  fr: i18n.t('common.langFrench', { lng: 'fr' }),
  de: i18n.t('common.langGerman', { lng: 'de' }),
  he: i18n.t('common.langHebrew', { lng: 'he' }),
  it: i18n.t('common.langItalian', { lng: 'it' }),
  ja: i18n.t('common.langJapanese', { lng: 'ja' }),
  ko: i18n.t('common.langKorean', { lng: 'ko' }),
  pl: i18n.t('common.langPolish', { lng: 'pl' }),
  pt_BR: i18n.t('common.langBrPortuguese', { lng: 'pt_BR' }),
  pt: i18n.t('common.langPortuguese', { lng: 'pt' }),
  ru: i18n.t('common.langRussian', { lng: 'ru' }),
  zh_CN: i18n.t('common.langSimplifiedChinese', { lng: 'zh_CN' }),
  es: i18n.t('common.langSpanish', { lng: 'es' }),
  uk: i18n.t('common.langUkranian', { lng: 'ua' }),
};

export default function LanguagePicker() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const language = useAppSelector(languageSelector);

  const renderLanguagePicker = () => {
    const languagesToRender: ReactNode[] = [];
    Object.keys(LANGUAGES).forEach((lang) => {
      const l = lang as keyof typeof LANGUAGES;
      languagesToRender.push(
        <IAIButton
          key={lang}
          isChecked={language === l}
          leftIcon={language === l ? <FaCheck /> : undefined}
          onClick={() => dispatch(languageChanged(l))}
          aria-label={LANGUAGES[l]}
          size="sm"
          minWidth="200px"
        >
          {LANGUAGES[l]}
        </IAIButton>
      );
    });

    return languagesToRender;
  };

  return (
    <IAIPopover
      triggerComponent={
        <IAIIconButton
          aria-label={t('common.languagePickerLabel')}
          tooltip={t('common.languagePickerLabel')}
          icon={<FaLanguage />}
          size="sm"
          variant="link"
          data-variant="link"
          fontSize={26}
        />
      }
    >
      <VStack>{renderLanguagePicker()}</VStack>
    </IAIPopover>
  );
}
