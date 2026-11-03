import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import InspectorDataTab from './InspectorDataTab';
import InspectorDetailsTab from './InspectorDetailsTab';
import InspectorOutputsTab from './InspectorOutputsTab';
import InspectorTemplateTab from './InspectorTemplateTab';

const WorkflowNodeInspectorPanel = () => {
  const { t } = useTranslation();
  return (
    <Tabs variant="enclosed" display="flex" flexDir="column" w="full" h="full">
      <TabList>
        <Tab>{t('common.details')}</Tab>
        <Tab>{t('common.outputs')}</Tab>
        <Tab>{t('common.data')}</Tab>
        <Tab>{t('common.template')}</Tab>
      </TabList>

      <TabPanels h="full" pt={2}>
        <TabPanel h="full" p={0}>
          <InspectorDetailsTab />
        </TabPanel>
        <TabPanel h="full" p={0}>
          <InspectorOutputsTab />
        </TabPanel>
        <TabPanel h="full" p={0}>
          <InspectorDataTab />
        </TabPanel>
        <TabPanel h="full" p={0}>
          <InspectorTemplateTab />
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
};

export default memo(WorkflowNodeInspectorPanel);
