import {
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@chakra-ui/react';
import { memo } from 'react';
import NodeDataInspector from './NodeDataInspector';
import NodeResultsInspector from './NodeResultsInspector';
import NodeTemplateInspector from './NodeTemplateInspector';

const InspectorPanel = () => {
  return (
    <Flex
      layerStyle="first"
      sx={{
        flexDir: 'column',
        w: 'full',
        h: 'full',
        borderRadius: 'base',
        p: 4,
        gap: 2,
      }}
    >
      <Tabs
        variant="line"
        sx={{ display: 'flex', flexDir: 'column', w: 'full', h: 'full' }}
      >
        <TabList>
          <Tab>Node Outputs</Tab>
          <Tab>Node Data</Tab>
          <Tab>Node Template</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <NodeResultsInspector />
          </TabPanel>
          <TabPanel>
            <NodeDataInspector />
          </TabPanel>
          <TabPanel>
            <NodeTemplateInspector />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(InspectorPanel);
