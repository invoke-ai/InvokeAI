import { ChakraProvider, extendTheme } from '@chakra-ui/react';
import { ReactNode, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { theme as invokeAITheme } from 'theme/theme';
import { RootState } from './store';
import { useAppSelector } from './storeHooks';

import { greenTeaThemeColors } from 'theme/colors/greenTea';
import { invokeAIThemeColors } from 'theme/colors/invokeAI';
import { lightThemeColors } from 'theme/colors/lightTheme';
import { oceanBlueColors } from 'theme/colors/oceanBlue';
import '@fontsource/inter/100.css';
import '@fontsource/inter/200.css';
import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/inter/800.css';
import '@fontsource/inter/900.css';

type ThemeLocaleProviderProps = {
  children: ReactNode;
};

const THEMES = {
  dark: invokeAIThemeColors,
  light: lightThemeColors,
  green: greenTeaThemeColors,
  ocean: oceanBlueColors,
};

function ThemeLocaleProvider({ children }: ThemeLocaleProviderProps) {
  const { i18n } = useTranslation();

  const currentTheme = useAppSelector(
    (state: RootState) => state.ui.currentTheme
  );

  const direction = i18n.dir();

  const theme = extendTheme({
    ...invokeAITheme,
    colors: THEMES[currentTheme as keyof typeof THEMES],
    direction,
  });

  useEffect(() => {
    document.body.dir = direction;
  }, [direction]);

  return <ChakraProvider theme={theme}>{children}</ChakraProvider>;
}

export default ThemeLocaleProvider;
