import {
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@chakra-ui/react';
import { memo } from 'react';
import GeneralTab from './workflow/GeneralTab';
import LinearTab from './workflow/LinearTab';
import WorkflowTab from './workflow/WorkflowTab';

const WorkflowPanel = () => {
  return (
    <Flex
      layerStyle="first"
      sx={{
        flexDir: 'column',
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
          <Tab>Linear</Tab>
          <Tab>Details</Tab>
          <Tab>JSON</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <LinearTab />
          </TabPanel>
          <TabPanel>
            <GeneralTab />
          </TabPanel>
          <TabPanel>
            <WorkflowTab />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(WorkflowPanel);
