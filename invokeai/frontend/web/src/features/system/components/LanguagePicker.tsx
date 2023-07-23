import {
  IconButton,
  Menu,
  MenuButton,
  MenuItemOption,
  MenuList,
  MenuOptionGroup,
  Tooltip,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { map } from 'lodash-es';
import { useTranslation } from 'react-i18next';
import { IoLanguage } from 'react-icons/io5';
import { LANGUAGES } from '../store/constants';
import { languageSelector } from '../store/systemSelectors';
import { languageChanged } from '../store/systemSlice';

export default function LanguagePicker() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const language = useAppSelector(languageSelector);

  return (
    <Menu closeOnSelect={false}>
      <Tooltip label={t('common.languagePickerLabel')} hasArrow>
        <MenuButton
          as={IconButton}
          icon={<IoLanguage />}
          variant="link"
          aria-label={t('common.languagePickerLabel')}
          fontSize={22}
          minWidth={8}
        />
      </Tooltip>
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
  );
}
