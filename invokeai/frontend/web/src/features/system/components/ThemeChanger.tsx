import {
  IconButton,
  Menu,
  MenuButton,
  MenuItemOption,
  MenuList,
  MenuOptionGroup,
  Tooltip,
} from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setCurrentTheme } from 'features/ui/store/uiSlice';
import i18n from 'i18n';
import { map } from 'lodash-es';
import { useTranslation } from 'react-i18next';
import { FaPalette } from 'react-icons/fa';

export const THEMES = {
  dark: i18n.t('common.darkTheme'),
  light: i18n.t('common.lightTheme'),
  green: i18n.t('common.greenTheme'),
  ocean: i18n.t('common.oceanTheme'),
};

export default function ThemeChanger() {
  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const currentTheme = useAppSelector(
    (state: RootState) => state.ui.currentTheme
  );

  return (
    <Menu closeOnSelect={false}>
      <Tooltip label={t('common.themeLabel')} hasArrow>
        <MenuButton
          as={IconButton}
          icon={<FaPalette />}
          variant="link"
          aria-label={t('common.themeLabel')}
          fontSize={20}
          minWidth={8}
        />
      </Tooltip>
      <MenuList>
        <MenuOptionGroup value={currentTheme}>
          {map(THEMES, (themeName, themeKey: keyof typeof THEMES) => (
            <MenuItemOption
              key={themeKey}
              value={themeKey}
              onClick={() => dispatch(setCurrentTheme(themeKey))}
            >
              {themeName}
            </MenuItemOption>
          ))}
        </MenuOptionGroup>
      </MenuList>
    </Menu>
  );
}
