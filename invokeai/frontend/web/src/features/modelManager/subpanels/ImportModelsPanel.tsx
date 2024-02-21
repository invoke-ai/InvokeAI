import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import AddModels from './AddModelsPanel/AddModels';
import ScanModels from './AddModelsPanel/ScanModels';

type AddModelTabs = 'add' | 'scan';

const ImportModelsPanel = () => {
  const [addModelTab, setAddModelTab] = useState<AddModelTabs>('add');
  const { t } = useTranslation();

  const handleClickAddTab = useCallback(() => setAddModelTab('add'), []);
  const handleClickScanTab = useCallback(() => setAddModelTab('scan'), []);

  return (
    <Flex flexDirection="column" gap={4} h="full">
      <ButtonGroup>
        <Button onClick={handleClickAddTab} isChecked={addModelTab === 'add'} size="sm" width="100%">
          {t('modelManager.addModel')}
        </Button>
        <Button onClick={handleClickScanTab} isChecked={addModelTab === 'scan'} size="sm" width="100%">
          {t('modelManager.scanForModels')}
        </Button>
      </ButtonGroup>

      {addModelTab === 'add' && <AddModels />}
      {addModelTab === 'scan' && <ScanModels />}
    </Flex>
  );
};

export default memo(ImportModelsPanel);
