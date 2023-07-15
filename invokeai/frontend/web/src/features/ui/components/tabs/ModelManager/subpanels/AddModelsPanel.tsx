import { ButtonGroup, Divider, Flex } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import AddModels from './AddModelsPanel/AddModels';
import ScanModels from './AddModelsPanel/ScanModels';

type AddModelTabs = 'add' | 'scan';

export default function AddModelsPanel() {
  const [addModelTab, setAddModelTab] = useState<AddModelTabs>('add');
  const { t } = useTranslation();

  return (
    <Flex flexDirection="column" gap={4}>
      <ButtonGroup isAttached>
        <IAIButton
          onClick={() => setAddModelTab('add')}
          isChecked={addModelTab == 'add'}
          size="sm"
        >
          {t('modelManager.addModel')}
        </IAIButton>
        <IAIButton
          onClick={() => setAddModelTab('scan')}
          isChecked={addModelTab == 'scan'}
          size="sm"
        >
          {t('modelManager.scanForModels')}
        </IAIButton>
      </ButtonGroup>

      <Divider />

      {addModelTab == 'add' && <AddModels />}
      {addModelTab == 'scan' && <ScanModels />}
    </Flex>
  );
}
