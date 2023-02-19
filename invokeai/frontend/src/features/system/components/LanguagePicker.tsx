import type { ReactNode } from 'react';

import { VStack } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { useTranslation } from 'react-i18next';
import { FaLanguage } from 'react-icons/fa';

export default function LanguagePicker() {
  const { t, i18n } = useTranslation();
  const LANGUAGES = {
    ar: t('common.langArabic', { lng: 'ar' }),
    nl: t('common.langDutch', { lng: 'nl' }),
    en: t('common.langEnglish', { lng: 'en' }),
    fr: t('common.langFrench', { lng: 'fr' }),
    de: t('common.langGerman', { lng: 'de' }),
    it: t('common.langItalian', { lng: 'it' }),
    ja: t('common.langJapanese', { lng: 'ja' }),
    pl: t('common.langPolish', { lng: 'pl' }),
    pt_Br: t('common.langBrPortuguese', { lng: 'pt_Br' }),
    ru: t('common.langRussian', { lng: 'ru' }),
    zh_Cn: t('common.langSimplifiedChinese', { lng: 'zh_Cn' }),
    es: t('common.langSpanish', { lng: 'es' }),
    uk: t('common.langUkranian', { lng: 'ua' }),
  };

  const renderLanguagePicker = () => {
    const languagesToRender: ReactNode[] = [];
    Object.keys(LANGUAGES).forEach((lang) => {
      languagesToRender.push(
        <IAIButton
          key={lang}
          data-selected={localStorage.getItem('i18nextLng') === lang}
          onClick={() => i18n.changeLanguage(lang)}
          className="modal-close-btn lang-select-btn"
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
      trigger="hover"
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
