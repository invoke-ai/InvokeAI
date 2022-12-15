import { useEffect } from 'react';
import { useColorMode, VStack } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { setCurrentTheme } from 'features/options/store/optionsSlice';
import IAIPopover from 'common/components/IAIPopover';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaCheck, FaPalette } from 'react-icons/fa';
import IAIButton from 'common/components/IAIButton';

const THEMES = ['dark', 'light', 'green'];

export default function ThemeChanger() {
  const { setColorMode, colorMode } = useColorMode();
  const dispatch = useAppDispatch();
  const currentTheme = useAppSelector(
    (state: RootState) => state.options.currentTheme
  );

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
          aria-label="Theme"
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
