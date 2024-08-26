import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { LoRACard } from 'features/lora/components/LoRACard';
import { memo } from 'react';

const selectLoRAsArray = createMemoizedSelector(selectLoRAsSlice, (loras) => loras.loras);

export const LoRAList = memo(() => {
  const lorasArray = useAppSelector(selectLoRAsArray);

  if (!lorasArray.length) {
    return null;
  }

  return (
    <Flex flexWrap="wrap" gap={2}>
      {lorasArray.map((lora) => (
        <LoRACard key={lora.id} lora={lora} />
      ))}
    </Flex>
  );
});

LoRAList.displayName = 'LoRAList';
