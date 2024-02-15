import { Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import WorkflowGeneralTab from './WorkflowGeneralTab';
import WorkflowJSONTab from './WorkflowJSONTab';
import WorkflowLinearTab from './WorkflowLinearTab';

const WorkflowPanel = () => {
  const { t } = useTranslation();
  return (
    <Flex layerStyle="first" flexDir="column" w="full" h="full" borderRadius="base" p={2} gap={2}>
      <Tabs variant="line" display="flex" w="full" h="full" flexDir="column">
        <TabList>
          <Tab>{t('common.details')}</Tab>
          <Tab>{t('common.linear')}</Tab>
          <Tab>JSON</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <WorkflowGeneralTab />
          </TabPanel>
          <TabPanel>
            <WorkflowLinearTab />
          </TabPanel>
          <TabPanel>
            <WorkflowJSONTab />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(WorkflowPanel);
