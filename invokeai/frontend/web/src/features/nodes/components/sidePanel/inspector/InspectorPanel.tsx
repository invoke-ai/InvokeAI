import {
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@chakra-ui/react';
import { memo } from 'react';
import InspectorDataTab from './InspectorDataTab';
import InspectorOutputsTab from './InspectorOutputsTab';
import InspectorTemplateTab from './InspectorTemplateTab';
// import InspectorDetailsTab from './InspectorDetailsTab';

const InspectorPanel = () => {
  return (
    <Flex
      layerStyle="first"
      sx={{
        flexDir: 'column',
        w: 'full',
        h: 'full',
        borderRadius: 'base',
        p: 2,
        gap: 2,
      }}
    >
      <Tabs
        variant="line"
        sx={{ display: 'flex', flexDir: 'column', w: 'full', h: 'full' }}
      >
        <TabList>
          {/* <Tab>Details</Tab> */}
          <Tab>Outputs</Tab>
          <Tab>Data</Tab>
          <Tab>Template</Tab>
        </TabList>

        <TabPanels>
          {/* <TabPanel>
            <InspectorDetailsTab />
          </TabPanel> */}
          <TabPanel>
            <InspectorOutputsTab />
          </TabPanel>
          <TabPanel>
            <InspectorDataTab />
          </TabPanel>
          <TabPanel>
            <InspectorTemplateTab />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(InspectorPanel);
