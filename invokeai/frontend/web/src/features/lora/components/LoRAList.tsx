import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { LoRACard } from 'features/lora/components/LoRACard';
import { selectLoraSlice } from 'features/lora/store/loraSlice';
import { map } from 'lodash-es';
import { memo } from 'react';

const selectLoRAsArray = createMemoizedSelector(selectLoraSlice, (lora) => map(lora.loras));

export const LoRAList = memo(() => {
  const lorasArray = useAppSelector(selectLoRAsArray);

  if (!lorasArray.length) {
    return null;
  }

  return (
    <Flex flexWrap="wrap" gap={2}>
      {lorasArray.map((lora) => (
        <LoRACard key={lora.key} lora={lora} />
      ))}
    </Flex>
  );
});

LoRAList.displayName = 'LoRAList';
