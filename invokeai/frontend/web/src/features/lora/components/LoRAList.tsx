import { Flex } from '@chakra-ui/layout';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { LoRACard } from 'features/lora/components/LoRACard';
import { map } from 'lodash-es';
import { memo } from 'react';

const selector = createMemoizedSelector(stateSelector, ({ lora }) => {
  return { lorasArray: map(lora.loras) };
});

export const LoRAList = memo(() => {
  const { lorasArray } = useAppSelector(selector);

  if (!lorasArray.length) {
    return null;
  }

  return (
    <Flex flexWrap="wrap" gap={2}>
      {lorasArray.map((lora) => (
        <LoRACard key={lora.model_name} lora={lora} />
      ))}
    </Flex>
  );
});

LoRAList.displayName = 'LoRAList';
