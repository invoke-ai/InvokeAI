import {
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { isInvocationNode } from 'features/nodes/types/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import InspectorDataTab from './InspectorDataTab';
import InspectorDetailsTab from './InspectorDetailsTab';
import InspectorNotesTab from './InspectorNotesTab';
import InspectorResultsTab from './InspectorResultsTab';
import InspectorTemplateTab from './InspectorTemplateTab';
import EditableNodeTitle from './details/EditableNodeTitle';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const lastSelectedNodeId =
      nodes.selectedNodes[nodes.selectedNodes.length - 1];

    const lastSelectedNode = nodes.nodes.find(
      (node) => node.id === lastSelectedNodeId
    );

    if (!isInvocationNode(lastSelectedNode)) {
      return;
    }

    return lastSelectedNode.id;
  },
  defaultSelectorOptions
);
const InspectorPanel = () => {
  const { t } = useTranslation();
  const nodeId = useAppSelector(selector);
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
      <EditableNodeTitle nodeId={nodeId} />
      <Tabs
        variant="line"
        sx={{ display: 'flex', flexDir: 'column', w: 'full', h: 'full' }}
      >
        <TabList>
          <Tab>{t('nodes.tabDetails')}</Tab>
          <Tab>{t('nodes.tabNotes')}</Tab>
          <Tab>{t('nodes.tabResults')}</Tab>
          <Tab>{t('nodes.tabData')}</Tab>
          <Tab>{t('nodes.tabTemplate')}</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <InspectorDetailsTab />
          </TabPanel>
          <TabPanel>
            <InspectorNotesTab />
          </TabPanel>
          <TabPanel>
            <InspectorResultsTab />
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
