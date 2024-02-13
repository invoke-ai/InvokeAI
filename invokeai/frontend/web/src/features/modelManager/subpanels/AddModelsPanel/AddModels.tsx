import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import AdvancedAddModels from './AdvancedAddModels';
import SimpleAddModels from './SimpleAddModels';

const AddModels = () => {
  const { t } = useTranslation();
  const [addModelMode, setAddModelMode] = useState<'simple' | 'advanced'>('simple');
  const handleAddModelSimple = useCallback(() => setAddModelMode('simple'), []);
  const handleAddModelAdvanced = useCallback(() => setAddModelMode('advanced'), []);
  return (
    <Flex flexDirection="column" width="100%" overflow="scroll" maxHeight={window.innerHeight - 250} gap={4}>
      <ButtonGroup>
        <Button size="sm" isChecked={addModelMode === 'simple'} onClick={handleAddModelSimple}>
          {t('common.simple')}
        </Button>
        <Button size="sm" isChecked={addModelMode === 'advanced'} onClick={handleAddModelAdvanced}>
          {t('common.advanced')}
        </Button>
      </ButtonGroup>
      <Flex p={4} borderRadius={4} bg="base.800">
        {addModelMode === 'simple' && <SimpleAddModels />}
        {addModelMode === 'advanced' && <AdvancedAddModels />}
      </Flex>
    </Flex>
  );
};

export default memo(AddModels);
