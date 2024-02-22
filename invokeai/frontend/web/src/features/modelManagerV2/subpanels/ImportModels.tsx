import { Box, Divider, Heading, Tab, Flex, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';

import { ImportQueue } from './AddModelPanel/ImportQueue';
import { SimpleImport } from './AddModelPanel/SimpleImport';
import { ScanModels } from './AddModelPanel/ScanModels/ScanModels';

export const ImportModels = () => {
  return (
    <Flex layerStyle="first" p={3} borderRadius="base" w="full" h="full" flexDir="column" gap={2}>
      <Box w="full" p={2}>
        <Heading fontSize="xl">Add Model</Heading>
      </Box>
      <Box layerStyle="second" borderRadius="base" w="full" h="50%" overflow="hidden">
        <Tabs variant="collapse" height="100%">
          <TabList>
            <Tab>Simple</Tab>
            <Tab>Advanced</Tab>
            <Tab>Scan</Tab>
          </TabList>
          <TabPanels p={3} height="100%">
            <TabPanel>
              <SimpleImport />
            </TabPanel>
            <TabPanel>Advanced Import Placeholder</TabPanel>
            <TabPanel height="100%">
              <ScanModels />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
      <Box layerStyle="second" borderRadius="base" w="full" h="50%">
        <ImportQueue />
      </Box>
    </Flex>
  );
};
