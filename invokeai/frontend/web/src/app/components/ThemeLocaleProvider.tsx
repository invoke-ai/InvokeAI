import '@fontsource-variable/inter';
import 'overlayscrollbars/overlayscrollbars.css';
import '@xyflow/react/dist/base.css';
import 'common/components/OverlayScrollbars/overlayscrollbars.css';

import { ChakraProvider, DarkMode, extendTheme, theme as baseTheme, TOAST_OPTIONS } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $direction } from 'app/hooks/useSyncLangDirection';
import type { ReactNode } from 'react';
import { memo, useMemo } from 'react';

type ThemeLocaleProviderProps = {
  children: ReactNode;
};

const buildTheme = (direction: 'ltr' | 'rtl') => {
  return extendTheme({
    ...baseTheme,
    direction,
    shadows: {
      ...baseTheme.shadows,
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
};

function ThemeLocaleProvider({ children }: ThemeLocaleProviderProps) {
  const direction = useStore($direction);
  const theme = useMemo(() => buildTheme(direction), [direction]);

  return (
    <ChakraProvider theme={theme} toastOptions={TOAST_OPTIONS}>
      <DarkMode>{children}</DarkMode>
    </ChakraProvider>
  );
}

export default memo(ThemeLocaleProvider);
