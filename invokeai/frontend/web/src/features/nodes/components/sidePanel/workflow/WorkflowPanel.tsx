import { Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { WorkflowBuilder } from 'features/nodes/components/sidePanel/builder/WorkflowBuilder';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import WorkflowGeneralTab from './WorkflowGeneralTab';
import WorkflowJSONTab from './WorkflowJSONTab';

const WorkflowFieldsLinearViewPanel = () => {
  const { t } = useTranslation();
  return (
    <Flex layerStyle="first" flexDir="column" w="full" h="full" borderRadius="base" p={2} gap={2}>
      <Tabs variant="line" display="flex" w="full" h="full" flexDir="column">
        <TabList>
          <Tab>{t('workflows.builder.builder')}</Tab>
          <Tab>{t('common.details')}</Tab>
          <Tab>JSON</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <WorkflowBuilder />
          </TabPanel>
          <TabPanel>
            <WorkflowGeneralTab />
          </TabPanel>
          <TabPanel>
            <WorkflowJSONTab />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(WorkflowFieldsLinearViewPanel);
