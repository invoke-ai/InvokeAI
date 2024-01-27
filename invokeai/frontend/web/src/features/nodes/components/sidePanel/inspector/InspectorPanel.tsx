import { Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import InspectorDataTab from './InspectorDataTab';
import InspectorDetailsTab from './InspectorDetailsTab';
import InspectorOutputsTab from './InspectorOutputsTab';
import InspectorTemplateTab from './InspectorTemplateTab';

const InspectorPanel = () => {
  const { t } = useTranslation();
  return (
    <Flex layerStyle="first" flexDir="column" w="full" h="full" borderRadius="base" p={2} gap={2}>
      <Tabs variant="line" display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('common.details')}</Tab>
          <Tab>{t('common.outputs')}</Tab>
          <Tab>{t('common.data')}</Tab>
          <Tab>{t('common.template')}</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <InspectorDetailsTab />
          </TabPanel>
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
