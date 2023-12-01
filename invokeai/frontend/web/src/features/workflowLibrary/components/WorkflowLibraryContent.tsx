import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@chakra-ui/react';
import WorkflowLibraryListWrapper from 'features/workflowLibrary/components/WorkflowLibraryListWrapper';
import WorkflowLibrarySystemList from 'features/workflowLibrary/components/WorkflowLibrarySystemList';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import WorkflowLibraryUserList from './WorkflowLibraryUserList';

const WorkflowLibraryContent = () => {
  const { t } = useTranslation();

  return (
    <Tabs w="full" h="full" isLazy>
      <TabList w="10rem" layerStyle="second" borderRadius="base" p={2}>
        <Tab>{t('workflows.user')}</Tab>
        <Tab>{t('workflows.system')}</Tab>
      </TabList>

      <TabPanels>
        <TabPanel>
          <WorkflowLibraryListWrapper>
            <WorkflowLibraryUserList />
          </WorkflowLibraryListWrapper>
        </TabPanel>
        <TabPanel>
          <WorkflowLibraryListWrapper>
            <WorkflowLibrarySystemList />
          </WorkflowLibraryListWrapper>
        </TabPanel>
      </TabPanels>
    </Tabs>
  );
};

export default memo(WorkflowLibraryContent);
