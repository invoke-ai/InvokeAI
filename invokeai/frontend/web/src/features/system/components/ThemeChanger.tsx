import { VStack } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import { setCurrentTheme } from 'features/ui/store/uiSlice';
import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCheck, FaPalette } from 'react-icons/fa';

export default function ThemeChanger() {
  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const currentTheme = useAppSelector(
    (state: RootState) => state.ui.currentTheme
  );

  const THEMES = {
    dark: t('common.darkTheme'),
    light: t('common.lightTheme'),
    green: t('common.greenTheme'),
    ocean: t('common.oceanTheme'),
  };

  const handleChangeTheme = (theme: string) => {
    dispatch(setCurrentTheme(theme));
  };

  const renderThemeOptions = () => {
    const themesToRender: ReactNode[] = [];

    Object.keys(THEMES).forEach((theme) => {
      themesToRender.push(
        <IAIButton
          sx={{
            width: 24,
          }}
          isChecked={currentTheme === theme}
          leftIcon={currentTheme === theme ? <FaCheck /> : undefined}
          size="sm"
          onClick={() => handleChangeTheme(theme)}
          key={theme}
        >
          {THEMES[theme as keyof typeof THEMES]}
        </IAIButton>
      );
    });

    return themesToRender;
  };

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label={t('common.themeLabel')}
          size="sm"
          variant="link"
          data-variant="link"
          fontSize={20}
          icon={<FaPalette />}
        />
      }
    >
      <VStack align="stretch">{renderThemeOptions()}</VStack>
    </IAIPopover>
  );
}
