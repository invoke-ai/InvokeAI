import { Flex } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useState } from 'react';
import AdvancedAddCheckpoint from './AdvancedAddCheckpoint';
import AdvancedAddDiffusers from './AdvancedAddDiffusers';

export const advancedAddModeData: SelectItem[] = [
  { label: 'Diffusers', value: 'diffusers' },
  { label: 'Checkpoint / Safetensors', value: 'checkpoint' },
];

export type ManualAddMode = 'diffusers' | 'checkpoint';

export default function AdvancedAddModels() {
  const [advancedAddMode, setAdvancedAddMode] =
    useState<ManualAddMode>('diffusers');

  return (
    <Flex flexDirection="column" gap={4} width="100%">
      <IAIMantineSelect
        label="Model Type"
        value={advancedAddMode}
        data={advancedAddModeData}
        onChange={(v) => {
          if (!v) {
            return;
          }
          setAdvancedAddMode(v as ManualAddMode);
        }}
      />

      <Flex
        sx={{
          p: 4,
          borderRadius: 4,
          bg: 'base.300',
          _dark: {
            bg: 'base.850',
          },
        }}
      >
        {advancedAddMode === 'diffusers' && <AdvancedAddDiffusers />}
        {advancedAddMode === 'checkpoint' && <AdvancedAddCheckpoint />}
      </Flex>
    </Flex>
  );
}
