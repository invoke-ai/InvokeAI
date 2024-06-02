import '@fontsource-variable/inter';
import 'overlayscrollbars/overlayscrollbars.css';

import { ChakraProvider, DarkMode, extendTheme, theme as _theme, TOAST_OPTIONS } from '@invoke-ai/ui-library';
import type { ReactNode } from 'react';
import { memo, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type ThemeLocaleProviderProps = {
  children: ReactNode;
};

function ThemeLocaleProvider({ children }: ThemeLocaleProviderProps) {
  const { i18n } = useTranslation();

  const direction = i18n.dir();

  const theme = useMemo(() => {
    return extendTheme({
      ..._theme,
      direction,
      shadows: {
        ..._theme.shadows,
        selectedForCompare:
          '0px 0px 0px 1px var(--invoke-colors-base-900), 0px 0px 0px 4px var(--invoke-colors-green-400)',
        hoverSelectedForCompare:
          '0px 0px 0px 1px var(--invoke-colors-base-900), 0px 0px 0px 4px var(--invoke-colors-green-300)',
      },
    });
  }, [direction]);

  useEffect(() => {
    document.body.dir = direction;
  }, [direction]);

  return (
    <ChakraProvider theme={theme} toastOptions={TOAST_OPTIONS}>
      <DarkMode>{children}</DarkMode>
    </ChakraProvider>
  );
}

export default memo(ThemeLocaleProvider);
