import { ChakraProvider } from '@chakra-ui/react';

import { system } from './theme/system';
import { AppToaster } from './workbench/components/ui/toaster';
import { ThemeController } from './workbench/ThemeController';
import { ModelsRuntime } from './workbench/widgets/models/ModelsRuntime';
import { WorkbenchProvider } from './workbench/WorkbenchContext';
import { WorkbenchRuntime } from './workbench/WorkbenchRuntime';
import { WorkbenchShell } from './workbench/WorkbenchShell';

export const App = () => (
  <ChakraProvider value={system}>
    <AppToaster />
    <WorkbenchProvider>
      <ThemeController />
      <WorkbenchRuntime />
      <ModelsRuntime />
      <WorkbenchShell />
    </WorkbenchProvider>
  </ChakraProvider>
);
