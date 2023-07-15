import { ButtonGroup, Flex } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import { useState } from 'react';
import AutoAddModels from './AutoAddModels';

export default function AddModels() {
  const [addModelMode, setAddModelMode] = useState<'simple' | 'advanced'>(
    'simple'
  );
  return (
    <Flex
      flexDirection="column"
      width="100%"
      overflow="scroll"
      maxHeight={window.innerHeight - 270}
      gap={4}
    >
      <ButtonGroup isAttached>
        <IAIButton
          size="sm"
          isChecked={addModelMode == 'simple'}
          onClick={() => setAddModelMode('simple')}
        >
          Simple
        </IAIButton>
        <IAIButton
          size="sm"
          isChecked={addModelMode == 'advanced'}
          onClick={() => setAddModelMode('advanced')}
        >
          Advanced
        </IAIButton>
      </ButtonGroup>
      <Flex
        sx={{
          p: 4,
          borderRadius: 4,
          background: 'base.200',
          _dark: { background: 'base.800' },
        }}
      >
        {addModelMode === 'simple' && <AutoAddModels />}
        {addModelMode === 'advanced' && null}
      </Flex>
    </Flex>
  );
}
