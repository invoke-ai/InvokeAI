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
        selected:
          'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-500), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
        hoverSelected:
          'inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-400), inset 0px 0px 0px 4px var(--invoke-colors-invokeBlue-800)',
        hoverUnselected:
          'inset 0px 0px 0px 2px var(--invoke-colors-invokeBlue-300), inset 0px 0px 0px 3px var(--invoke-colors-invokeBlue-800)',
        selectedForCompare:
          'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-300), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
        hoverSelectedForCompare:
          'inset 0px 0px 0px 3px var(--invoke-colors-invokeGreen-200), inset 0px 0px 0px 4px var(--invoke-colors-invokeGreen-800)',
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
