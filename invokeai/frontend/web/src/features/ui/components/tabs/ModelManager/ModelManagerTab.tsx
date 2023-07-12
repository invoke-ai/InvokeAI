import {
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  useColorMode,
} from '@chakra-ui/react';
import i18n from 'i18n';
import { ReactNode, memo } from 'react';
import { mode } from 'theme/util/mode';
import AddModelsPanel from './subpanels/AddModelsPanel';
import MergeModelsPanel from './subpanels/MergeModelsPanel';
import ModelManagerPanel from './subpanels/ModelManagerPanel';

type ModelManagerTabName = 'modelManager' | 'addModels' | 'mergeModels';

type ModelManagerTabInfo = {
  id: ModelManagerTabName;
  label: string;
  content: ReactNode;
};

const modelManagerTabs: ModelManagerTabInfo[] = [
  {
    id: 'modelManager',
    label: i18n.t('modelManager.modelManager'),
    content: <ModelManagerPanel />,
  },
  {
    id: 'addModels',
    label: i18n.t('modelManager.addModel'),
    content: <AddModelsPanel />,
  },
  {
    id: 'mergeModels',
    label: i18n.t('modelManager.mergeModels'),
    content: <MergeModelsPanel />,
  },
];

const ModelManagerTab = () => {
  const { colorMode } = useColorMode();

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
          color: mode('base.900', 'base.400')(colorMode),
          flexDirection: 'row',
          borderBottomWidth: 2,
          borderColor: mode('accent.300', 'accent.600')(colorMode),
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
