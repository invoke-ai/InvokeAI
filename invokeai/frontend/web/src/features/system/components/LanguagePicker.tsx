import type { ReactNode } from 'react';

import { VStack } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { useTranslation } from 'react-i18next';
import { FaCheck, FaLanguage } from 'react-icons/fa';

export default function LanguagePicker() {
  const { t, i18n } = useTranslation();
  const LANGUAGES = {
    ar: t('common.langArabic', { lng: 'ar' }),
    nl: t('common.langDutch', { lng: 'nl' }),
    en: t('common.langEnglish', { lng: 'en' }),
    fr: t('common.langFrench', { lng: 'fr' }),
    de: t('common.langGerman', { lng: 'de' }),
    he: t('common.langHebrew', { lng: 'he' }),
    it: t('common.langItalian', { lng: 'it' }),
    ja: t('common.langJapanese', { lng: 'ja' }),
    ko: t('common.langKorean', { lng: 'ko' }),
    pl: t('common.langPolish', { lng: 'pl' }),
    pt_BR: t('common.langBrPortuguese', { lng: 'pt_BR' }),
    pt: t('common.langPortuguese', { lng: 'pt' }),
    ru: t('common.langRussian', { lng: 'ru' }),
    zh_CN: t('common.langSimplifiedChinese', { lng: 'zh_CN' }),
    es: t('common.langSpanish', { lng: 'es' }),
    uk: t('common.langUkranian', { lng: 'ua' }),
  };

  const renderLanguagePicker = () => {
    const languagesToRender: ReactNode[] = [];
    Object.keys(LANGUAGES).forEach((lang) => {
      languagesToRender.push(
        <IAIButton
          key={lang}
          isChecked={localStorage.getItem('i18nextLng') === lang}
          leftIcon={
            localStorage.getItem('i18nextLng') === lang ? (
              <FaCheck />
            ) : undefined
          }
          onClick={() => i18n.changeLanguage(lang)}
          aria-label={LANGUAGES[lang as keyof typeof LANGUAGES]}
          size="sm"
          minWidth="200px"
        >
          {LANGUAGES[lang as keyof typeof LANGUAGES]}
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
