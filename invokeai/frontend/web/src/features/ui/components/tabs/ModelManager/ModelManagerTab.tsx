import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@chakra-ui/react';
import i18n from 'i18n';
import { ReactNode, memo } from 'react';
import ImportModelsPanel from './subpanels/ImportModelsPanel';
import MergeModelsPanel from './subpanels/MergeModelsPanel';
import ModelManagerPanel from './subpanels/ModelManagerPanel';

type ModelManagerTabName = 'modelManager' | 'importModels' | 'mergeModels';

type ModelManagerTabInfo = {
  id: ModelManagerTabName;
  label: string;
  content: ReactNode;
};

const tabs: ModelManagerTabInfo[] = [
  {
    id: 'modelManager',
    label: i18n.t('modelManager.modelManager'),
    content: <ModelManagerPanel />,
  },
  {
    id: 'importModels',
    label: i18n.t('modelManager.importModels'),
    content: <ImportModelsPanel />,
  },
  {
    id: 'mergeModels',
    label: i18n.t('modelManager.mergeModels'),
    content: <MergeModelsPanel />,
  },
];

const ModelManagerTab = () => {
  return (
    <Tabs
      isLazy
      variant="line"
      layerStyle="first"
      sx={{ w: 'full', h: 'full', p: 4, gap: 4, borderRadius: 'base' }}
    >
      <TabList>
        {tabs.map((tab) => (
          <Tab sx={{ borderTopRadius: 'base' }} key={tab.id}>
            {tab.label}
          </Tab>
        ))}
      </TabList>
      <TabPanels sx={{ w: 'full', h: 'full' }}>
        {tabs.map((tab) => (
          <TabPanel sx={{ w: 'full', h: 'full' }} key={tab.id}>
            {tab.content}
          </TabPanel>
        ))}
      </TabPanels>
    </Tabs>
  );
};

export default memo(ModelManagerTab);
