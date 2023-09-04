import {
  ChakraProvider,
  createLocalStorageManager,
  extendTheme,
} from '@chakra-ui/react';
import { ReactNode, memo, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { theme as invokeAITheme } from 'theme/theme';

import '@fontsource-variable/inter';
import { MantineProvider } from '@mantine/core';
import 'overlayscrollbars/overlayscrollbars.css';
import 'theme/css/overlayscrollbars.css';
import { useMantineTheme } from 'mantine-theme/theme';

type ThemeLocaleProviderProps = {
  children: ReactNode;
};

const manager = createLocalStorageManager('@@invokeai-color-mode');

function ThemeLocaleProvider({ children }: ThemeLocaleProviderProps) {
  const { i18n } = useTranslation();

  const direction = i18n.dir();

  const theme = useMemo(() => {
    return extendTheme({
      ...invokeAITheme,
      direction,
    });
  }, [direction]);

  useEffect(() => {
    document.body.dir = direction;
  }, [direction]);

  const mantineTheme = useMantineTheme();

  return (
    <MantineProvider theme={mantineTheme}>
      <ChakraProvider theme={theme} colorModeManager={manager}>
        {children}
      </ChakraProvider>
    </MantineProvider>
  );
}

export default memo(ThemeLocaleProvider);
