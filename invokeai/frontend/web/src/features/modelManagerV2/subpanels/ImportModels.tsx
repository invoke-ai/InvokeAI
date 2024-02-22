import { Box, Divider, Heading, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';

import { ImportQueue } from './AddModelPanel/ImportQueue';
import { SimpleImport } from './AddModelPanel/SimpleImport';

export const ImportModels = () => {
  return (
    <Box layerStyle="first" p={3} borderRadius="base" w="full" h="full">
      <Box w="full" p={4}>
        <Heading fontSize="xl">Add Model</Heading>
      </Box>
      <Box layerStyle="second" borderRadius="base" w="full" h="100vh">
        <Tabs variant="collapse">
          <TabList>
            <Tab>Simple</Tab>
            <Tab>Advanced</Tab>
            <Tab>Scan</Tab>
          </TabList>
          <TabPanels p={3}>
            <TabPanel>
              <SimpleImport />
            </TabPanel>
            <TabPanel>Advanced Import Placeholder</TabPanel>
            <TabPanel>Scan Models Placeholder</TabPanel>
          </TabPanels>
        </Tabs>

        <Divider mt="5" mb="3" />
        <ImportQueue />
      </Box>
    </Box>
  );
};
