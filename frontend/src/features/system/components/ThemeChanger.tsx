import { useEffect } from 'react';
import { useColorMode, VStack } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { setCurrentTheme } from 'features/options/store/optionsSlice';
import IAIPopover from 'common/components/IAIPopover';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaCheck, FaPalette } from 'react-icons/fa';
import IAIButton from 'common/components/IAIButton';
import { useTranslation } from 'react-i18next';

export default function ThemeChanger() {
  const { t } = useTranslation();
  const { setColorMode, colorMode } = useColorMode();
  const dispatch = useAppDispatch();
  const currentTheme = useAppSelector(
    (state: RootState) => state.options.currentTheme
  );

  const THEMES = [
    t('common:darkTheme'),
    t('common:lightTheme'),
    t('common:greenTheme'),
  ];

  useEffect(() => {
    // syncs the redux store theme to the chakra's theme on startup and when
    // setCurrentTheme is dispatched
    if (colorMode !== currentTheme) {
      setColorMode(currentTheme);
    }
  }, [setColorMode, colorMode, currentTheme]);

  const handleChangeTheme = (theme: string) => {
    dispatch(setCurrentTheme(theme));
  };

  return (
    <IAIPopover
      trigger="hover"
      triggerComponent={
        <IAIIconButton
          aria-label={t('common:themeLabel')}
          size={'sm'}
          variant="link"
          data-variant="link"
          fontSize={20}
          icon={<FaPalette />}
        />
      }
    >
      <VStack align={'stretch'}>
        {THEMES.map((theme) => (
          <IAIButton
            style={{
              width: '6rem',
            }}
            leftIcon={currentTheme === theme ? <FaCheck /> : undefined}
            size={'sm'}
            onClick={() => handleChangeTheme(theme)}
            key={theme}
          >
            {theme.charAt(0).toUpperCase() + theme.slice(1)}
          </IAIButton>
        ))}
      </VStack>
    </IAIPopover>
  );
}
