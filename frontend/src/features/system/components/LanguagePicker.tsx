import type { ReactNode } from 'react';

import { VStack } from '@chakra-ui/react';
import IAIPopover from 'common/components/IAIPopover';
import React from 'react';
import IAIButton from 'common/components/IAIButton';
import { useTranslation } from 'react-i18next';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaLanguage } from 'react-icons/fa';

export default function LanguagePicker() {
  const { t, i18n } = useTranslation();

  const LANGUAGES = {
    en: t('common:langEnglish'),
    ru: t('common:langRussian'),
    it: t('common:langItalian'),
    pt_br: t('common:langBrPortuguese'),
    de: t('common:langGerman'),
    pl: t('common:langPolish'),
    zh_cn: t('common:langSimplifiedChinese'),
    es: t('common:langSpanish'),
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
          tooltip={LANGUAGES[lang as keyof typeof LANGUAGES]}
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
          aria-label={t('common:languagePickerLabel')}
          tooltip={t('common:languagePickerLabel')}
          icon={<FaLanguage />}
          size={'sm'}
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
