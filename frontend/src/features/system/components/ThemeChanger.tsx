import { useColorMode, VStack } from '@chakra-ui/react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { setCurrentTheme } from 'features/options/store/optionsSlice';
import IAIPopover from 'common/components/IAIPopover';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaCheck, FaPalette } from 'react-icons/fa';
import IAIButton from 'common/components/IAIButton';

const THEMES = ['dark', 'light', 'green'];

export default function ThemeChanger() {
  const { setColorMode } = useColorMode();
  const dispatch = useAppDispatch();
  const currentTheme = useAppSelector(
    (state: RootState) => state.options.currentTheme
  );

  const handleChangeTheme = (theme: string) => {
    setColorMode(theme);
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
