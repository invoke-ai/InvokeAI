import { Button } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useStagingAreaContext } from 'features/controlLayers/components/SimpleSession/context';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useMemo } from 'react';

export const StagingAreaToolbarImageCountButton = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const ctx = useStagingAreaContext();
  const selectedItem = useStore(ctx.$selectedItem);
  const itemCount = useStore(ctx.$itemCount);

  const counterText = useMemo(() => {
    if (itemCount > 0 && selectedItem !== null) {
      return `${selectedItem.index + 1} of ${itemCount}`;
    } else {
      return `0 of 0`;
    }
  }, [itemCount, selectedItem]);

  return (
    <Button colorScheme="base" pointerEvents="none" minW={28} isDisabled={!shouldShowStagedImage}>
      {counterText}
    </Button>
  );
});

StagingAreaToolbarImageCountButton.displayName = 'StagingAreaToolbarImageCountButton';
