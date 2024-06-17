import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasV2Slice } from 'features/controlLayers/store/canvasV2Slice';
import { LoRACard } from 'features/lora/components/LoRACard';
import { memo } from 'react';

const selectLoRAsArray = createMemoizedSelector(selectCanvasV2Slice, (canvasV2) => canvasV2.loras);

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
