import { Button } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { memo, useMemo } from 'react';

export const StagingAreaToolbarImageCountButton = memo(() => {
  const ctx = useCanvasSessionContext();
  const selectItemIndex = useStore(ctx.$selectedItemIndex);
  const itemCount = useStore(ctx.$itemCount);

  const counterText = useMemo(() => {
    if (itemCount > 0 && selectItemIndex !== null) {
      return `${selectItemIndex + 1} of ${itemCount}`;
    } else {
      return `0 of 0`;
    }
  }, [itemCount, selectItemIndex]);

  return (
    <Button colorScheme="base" pointerEvents="none" minW={28}>
      {counterText}
    </Button>
  );
});

StagingAreaToolbarImageCountButton.displayName = 'StagingAreaToolbarImageCountButton';
