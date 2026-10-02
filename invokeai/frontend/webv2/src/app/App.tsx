import { AppProviders } from '@app/AppProviders';
import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster } from '@platform/ui/toaster';
import { RouterProvider } from '@tanstack/react-router';
import { system } from '@theme/system';
import { useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { lazy, Suspense } from 'react';

import { FeatureHintsAdapterProvider } from './FeatureHintsProvider';
import { I18nController } from './I18nController';
import { router } from './router';
import { ThemeController } from './ThemeController';

const AlphaNoticeDialog = lazy(() =>
  import('@workbench/shell/AlphaNoticeDialog').then((module) => ({ default: module.AlphaNoticeDialog }))
);

/** Load the alpha notice only for accounts that have not dismissed it. */
const AlphaNoticeGate = () => {
  const isDue = useWorkbenchSettingsSelector(
    (snapshot) => snapshot.status === 'ready' && !snapshot.preferences.alphaNoticeAcknowledged
  );

  return isDue ? (
    <Suspense fallback={null}>
      <AlphaNoticeDialog />
    </Suspense>
  ) : null;
};

export const App = () => (
  <AppProviders>
    <ChakraProvider value={system}>
      <ThemeController />
      <I18nController />
      <AppToaster />
      <AlphaNoticeGate />
      <FeatureHintsAdapterProvider>
        <RouterProvider router={router} />
      </FeatureHintsAdapterProvider>
    </ChakraProvider>
  </AppProviders>
);
