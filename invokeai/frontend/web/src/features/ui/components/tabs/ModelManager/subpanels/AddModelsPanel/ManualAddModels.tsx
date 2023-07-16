import { Flex } from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useState } from 'react';
import ManualAddCheckpoint from './ManualAddCheckpoint';
import ManualAddDiffusers from './ManualAddDiffusers';

const manualAddModeData: SelectItem[] = [
  { label: 'Diffusers', value: 'diffusers' },
  { label: 'Checkpoint / Safetensors', value: 'checkpoint' },
];

type ManualAddMode = 'diffusers' | 'checkpoint';

export default function ManualAddModels() {
  const [manualAddMode, setManualAddMode] =
    useState<ManualAddMode>('diffusers');

  return (
    <Flex flexDirection="column" gap={4} width="100%">
      <IAIMantineSelect
        label="Model Type"
        value={manualAddMode}
        data={manualAddModeData}
        onChange={(v) => {
          if (!v) return;
          setManualAddMode(v as ManualAddMode);
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
        {manualAddMode === 'diffusers' && <ManualAddDiffusers />}
        {manualAddMode === 'checkpoint' && <ManualAddCheckpoint />}
      </Flex>
    </Flex>
  );
}
