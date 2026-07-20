import { AppProviders } from '@app/AppProviders';
import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster } from '@platform/ui/toaster';
import { RouterProvider } from '@tanstack/react-router';
import { system } from '@theme/system';

import { I18nController } from './I18nController';
import { router } from './router';
import { ThemeController } from './ThemeController';

export const App = () => (
  <AppProviders>
    <ChakraProvider value={system}>
      <ThemeController />
      <I18nController />
      <AppToaster />
      <RouterProvider router={router} />
    </ChakraProvider>
  </AppProviders>
);
