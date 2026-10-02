import { AppProviders } from '@app/AppProviders';
import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster } from '@platform/ui/toaster';
import { RouterProvider } from '@tanstack/react-router';
import { system } from '@theme/system';
import { useWorkbenchSettingsSelector } from '@workbench/settings/store';
import { useWhatsNew } from '@workbench/shell/useWhatsNew';
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

const WhatsNewDialog = lazy(() =>
  import('@workbench/shell/WhatsNewDialog').then((module) => ({ default: module.WhatsNewDialog }))
);

/** Load the What's New notes when a new version has not been seen yet, or when the app menu asks for them. */
const WhatsNewGate = () => {
  const { isOpen } = useWhatsNew();

  return isOpen ? (
    <Suspense fallback={null}>
      <WhatsNewDialog />
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
      <WhatsNewGate />
      <FeatureHintsAdapterProvider>
        <RouterProvider router={router} />
      </FeatureHintsAdapterProvider>
    </ChakraProvider>
  </AppProviders>
);
