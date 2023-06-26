import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@chakra-ui/react';
import i18n from 'i18n';
import { ReactNode, memo } from 'react';
import AddModelsPanel from './subpanels/AddModelsPanel';
import MergeModelsPanel from './subpanels/MergeModelsPanel';
import ModelManagerPanel from './subpanels/ModelManagerPanel';

type ModelManagerTabName = 'modelmanager' | 'add_models' | 'merge_models';

type ModelManagerTabInfo = {
  id: ModelManagerTabName;
  label: string;
  content: ReactNode;
};

const modelManagerTabs: ModelManagerTabInfo[] = [
  {
    id: 'modelmanager',
    label: i18n.t('modelManager.modelManager'),
    content: <ModelManagerPanel />,
  },
  {
    id: 'add_models',
    label: i18n.t('modelManager.addModel'),
    content: <AddModelsPanel />,
  },
  {
    id: 'merge_models',
    label: i18n.t('modelManager.mergeModels'),
    content: <MergeModelsPanel />,
  },
];

const ModelManagerTab = () => {
  const renderTabsList = () => {
    const modelManagerTabListsToRender: ReactNode[] = [];
    modelManagerTabs.forEach((modelManagerTab) => {
      modelManagerTabListsToRender.push(
        <Tab key={modelManagerTab.id}>{modelManagerTab.label}</Tab>
      );
    });

    return (
      <TabList
        sx={{
          w: '100%',
          color: 'base.200',
          flexDirection: 'row',
          borderBottomWidth: 2,
          borderColor: 'accent.700',
        }}
      >
        {modelManagerTabListsToRender}
      </TabList>
    );
  };

  const renderTabPanels = () => {
    const modelManagerTabPanelsToRender: ReactNode[] = [];
    modelManagerTabs.forEach((modelManagerTab) => {
      modelManagerTabPanelsToRender.push(
        <TabPanel key={modelManagerTab.id}>{modelManagerTab.content}</TabPanel>
      );
    });

    return <TabPanels sx={{ p: 2 }}>{modelManagerTabPanelsToRender}</TabPanels>;
  };

  return (
    <Tabs
      isLazy
      variant="invokeAI"
      sx={{ w: 'full', h: 'full', p: 2, gap: 4, flexDirection: 'column' }}
    >
      {renderTabsList()}
      {renderTabPanels()}
    </Tabs>
  );
};

export default memo(ModelManagerTab);
