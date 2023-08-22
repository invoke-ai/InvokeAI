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
import NodeTemplateInspector from './NodeTemplateInspector';

const InspectorPanel = () => {
  return (
    <Flex
      layerStyle="first"
      sx={{
        w: 'full',
        h: 'full',
        borderRadius: 'base',
        p: 4,
      }}
    >
      <Tabs
        variant="line"
        sx={{ display: 'flex', flexDir: 'column', w: 'full', h: 'full' }}
      >
        <TabList>
          <Tab>Node Template</Tab>
          <Tab>Node Data</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <NodeTemplateInspector />
          </TabPanel>
          <TabPanel>
            <NodeDataInspector />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(InspectorPanel);
