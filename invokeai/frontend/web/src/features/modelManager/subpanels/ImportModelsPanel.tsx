import { Flex } from '@chakra-ui/react';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import AddModels from './AddModelsPanel/AddModels';
import ScanModels from './AddModelsPanel/ScanModels';

type AddModelTabs = 'add' | 'scan';

export default function ImportModelsPanel() {
  const [addModelTab, setAddModelTab] = useState<AddModelTabs>('add');
  const { t } = useTranslation();

  const handleClickAddTab = useCallback(() => setAddModelTab('add'), []);
  const handleClickScanTab = useCallback(() => setAddModelTab('scan'), []);

  return (
    <Flex flexDirection="column" gap={4}>
      <InvButtonGroup>
        <InvButton
          onClick={handleClickAddTab}
          isChecked={addModelTab == 'add'}
          size="sm"
          width="100%"
        >
          {t('modelManager.addModel')}
        </InvButton>
        <InvButton
          onClick={handleClickScanTab}
          isChecked={addModelTab == 'scan'}
          size="sm"
          width="100%"
        >
          {t('modelManager.scanForModels')}
        </InvButton>
      </InvButtonGroup>

      {addModelTab == 'add' && <AddModels />}
      {addModelTab == 'scan' && <ScanModels />}
    </Flex>
  );
}
