import { Spacer, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { WorkflowBuilder } from 'features/nodes/components/sidePanel/builder/WorkflowBuilder';
import { StartPublishFlowButton } from 'features/nodes/components/sidePanel/workflow/PublishWorkflowPanelContent';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import WorkflowGeneralTab from './WorkflowGeneralTab';
import WorkflowJSONTab from './WorkflowJSONTab';

const WorkflowFieldsLinearViewPanel = () => {
  const { t } = useTranslation();
  const publishWorkflowIsEnabled = useFeatureStatus('publishWorkflow');
  return (
    <Tabs variant="enclosed" display="flex" w="full" h="full" flexDir="column">
      <TabList>
        <Tab>{t('workflows.builder.builder')}</Tab>
        <Tab>{t('common.details')}</Tab>
        <Tab>JSON</Tab>
        <Spacer />
        {publishWorkflowIsEnabled && <StartPublishFlowButton />}
      </TabList>

      <TabPanels h="full" pt={2}>
        <TabPanel h="full" p={0}>
          <WorkflowBuilder />
        </TabPanel>
        <TabPanel h="full" p={0}>
          <WorkflowGeneralTab />
        </TabPanel>
        <TabPanel h="full" p={0}>
          <WorkflowJSONTab />
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
};

export default memo(WorkflowFieldsLinearViewPanel);
