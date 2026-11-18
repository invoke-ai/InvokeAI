import { AppProviders } from '@app/AppProviders';
import { ChakraProvider } from '@chakra-ui/react';
import { useExitPresence } from '@platform/react/useExitRetainedValue';
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

const BetaNoticeDialog = lazy(() =>
  import('@workbench/shell/BetaNoticeDialog').then((module) => ({ default: module.BetaNoticeDialog }))
);

/** Load the beta notice only for accounts that have not dismissed it. */
const BetaNoticeGate = () => {
  const isDue = useWorkbenchSettingsSelector(
    (snapshot) => snapshot.status === 'ready' && !snapshot.preferences.betaNoticeAcknowledged
  );
  // Stays mounted through the close animation instead of vanishing on dismissal.
  const dialog = useExitPresence(isDue);

  return dialog.isMounted ? (
    <Suspense fallback={null}>
      <BetaNoticeDialog onExitComplete={dialog.release} />
    </Suspense>
  ) : null;
};

const WhatsNewDialog = lazy(() =>
  import('@workbench/shell/WhatsNewDialog').then((module) => ({ default: module.WhatsNewDialog }))
);

/** Load the What's New notes when a new version has not been seen yet, or when the app menu asks for them. */
const WhatsNewGate = () => {
  const { isOpen } = useWhatsNew();
  // Stays mounted through the close animation instead of vanishing with the open state.
  const dialog = useExitPresence(isOpen);

  return dialog.isMounted ? (
    <Suspense fallback={null}>
      <WhatsNewDialog onExitComplete={dialog.release} />
    </Suspense>
  ) : null;
};

export const App = () => (
  <AppProviders>
    <ChakraProvider value={system}>
      <ThemeController />
      <I18nController />
      <AppToaster />
      <BetaNoticeGate />
      <WhatsNewGate />
      <FeatureHintsAdapterProvider>
        <RouterProvider router={router} />
      </FeatureHintsAdapterProvider>
    </ChakraProvider>
  </AppProviders>
);
