import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import ImportModelsPanel from 'features/modelManager/subpanels/ImportModelsPanel';
import MergeModelsPanel from 'features/modelManager/subpanels/MergeModelsPanel';
import ModelManagerPanel from 'features/modelManager/subpanels/ModelManagerPanel';
import ModelManagerSettingsPanel from 'features/modelManager/subpanels/ModelManagerSettingsPanel';
import type { ReactNode } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type ModelManagerTabName = 'modelManager' | 'importModels' | 'mergeModels' | 'settings';

type ModelManagerTabInfo = {
  id: ModelManagerTabName;
  label: string;
  content: ReactNode;
};

const ModelManagerTab = () => {
  const { t } = useTranslation();

  const tabs: ModelManagerTabInfo[] = useMemo(
    () => [
      {
        id: 'modelManager',
        label: t('modelManager.modelManager'),
        content: <ModelManagerPanel />,
      },
      {
        id: 'importModels',
        label: t('modelManager.importModels'),
        content: <ImportModelsPanel />,
      },
      {
        id: 'mergeModels',
        label: t('modelManager.mergeModels'),
        content: <MergeModelsPanel />,
      },
      {
        id: 'settings',
        label: t('modelManager.settings'),
        content: <ModelManagerSettingsPanel />,
      },
    ],
    [t]
  );
  return (
    <Tabs isLazy variant="line" layerStyle="first" w="full" h="full" p={4} gap={4} borderRadius="base">
      <TabList>
        {tabs.map((tab) => (
          <Tab borderTopRadius="base" key={tab.id}>
            {tab.label}
          </Tab>
        ))}
      </TabList>
      <TabPanels w="full" h="full">
        {tabs.map((tab) => (
          <TabPanel w="full" h="full" key={tab.id}>
            {tab.content}
          </TabPanel>
        ))}
      </TabPanels>
    </Tabs>
  );
};

export default memo(ModelManagerTab);
