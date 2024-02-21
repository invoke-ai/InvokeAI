import { Flex, Heading, Tab, TabList, TabPanel, TabPanels, Tabs, Box, Button } from '@invoke-ai/ui-library';
import ImportModelsPanel from 'features/modelManager/subpanels/ImportModelsPanel';
import MergeModelsPanel from 'features/modelManager/subpanels/MergeModelsPanel';
import ModelManagerPanel from 'features/modelManager/subpanels/ModelManagerPanel';
import ModelManagerSettingsPanel from 'features/modelManager/subpanels/ModelManagerSettingsPanel';
import type { ReactNode } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { SyncModelsIconButton } from '../../../modelManager/components/SyncModels/SyncModelsIconButton';
import { ModelManager } from '../../../modelManagerV2/subpanels/ModelManager';
import { ModelPane } from '../../../modelManagerV2/subpanels/ModelPane';

type ModelManagerTabName = 'modelManager' | 'importModels' | 'mergeModels' | 'settings';

const ModelManagerTab = () => {
  const { t } = useTranslation();

  return (
    <Flex w="full" h="full" gap="2">
      <ModelManager />
      <ModelPane />
    </Flex>
  );
};

export default memo(ModelManagerTab);
