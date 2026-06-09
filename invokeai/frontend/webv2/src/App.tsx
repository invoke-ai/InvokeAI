import { ChakraProvider } from '@chakra-ui/react';

import { system } from './theme/system';
import { WorkbenchProvider } from './workbench/WorkbenchContext';
import { WorkbenchShell } from './workbench/WorkbenchShell';

export const App = () => (
  <ChakraProvider value={system}>
    <WorkbenchProvider>
      <WorkbenchShell />
    </WorkbenchProvider>
  </ChakraProvider>
);
