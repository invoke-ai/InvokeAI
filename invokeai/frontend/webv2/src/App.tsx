import { ChakraProvider } from '@chakra-ui/react';
import { RouterProvider } from '@tanstack/react-router';

import { router } from './router';
import { system } from './theme/system';
import { AppToaster } from './workbench/components/ui/toaster';

export const App = () => (
  <ChakraProvider value={system}>
    <AppToaster />
    <RouterProvider router={router} />
  </ChakraProvider>
);
