import {
  Menu,
  MenuButton,
  MenuItemOption,
  MenuList,
  MenuOptionGroup,
  Tooltip,
} from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { useTranslation } from 'react-i18next';
import { FaLanguage } from 'react-icons/fa';
import i18n from 'i18n';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { languageSelector } from '../store/systemSelectors';
import { languageChanged } from '../store/systemSlice';
import { map } from 'lodash-es';

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

  return (
    <Tooltip title={t('common.languagePickerLabel')}>
      <Menu closeOnSelect={false}>
        <MenuButton
          as={IAIIconButton}
          icon={<FaLanguage />}
          variant="link"
          aria-label={t('common.languagePickerLabel')}
        />
        <MenuList>
          <MenuOptionGroup value={language}>
            {map(LANGUAGES, (languageName, l: keyof typeof LANGUAGES) => (
              <MenuItemOption
                key={l}
                value={l}
                onClick={() => dispatch(languageChanged(l))}
              >
                {languageName}
              </MenuItemOption>
            ))}
          </MenuOptionGroup>
        </MenuList>
      </Menu>
    </Tooltip>
  );
}
