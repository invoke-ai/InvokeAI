import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectLoRAsSlice } from 'features/controlLayers/store/lorasSlice';
import { LoRACard } from 'features/lora/components/LoRACard';
import { memo } from 'react';

const selectLoRAIds = createMemoizedSelector(selectLoRAsSlice, (loras) => loras.loras.map(({ id }) => id));

export const LoRAList = memo(() => {
  const ids = useAppSelector(selectLoRAIds);

  if (!ids.length) {
    return null;
  }

  return (
    <Flex flexWrap="wrap" gap={2}>
      {ids.map((id) => (
        <LoRACard key={id} id={id} />
      ))}
    </Flex>
  );
});

LoRAList.displayName = 'LoRAList';
