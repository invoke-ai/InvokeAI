import { Button } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectImageCount, selectStagedImageIndex } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { memo, useMemo } from 'react';

export const StagingAreaToolbarImageCountButton = memo(() => {
  const index = useAppSelector(selectStagedImageIndex);
  const imageCount = useAppSelector(selectImageCount);

  const counterText = useMemo(() => {
    if (imageCount > 0) {
      return `${(index ?? 0) + 1} of ${imageCount}`;
    } else {
      return `0 of 0`;
    }
  }, [imageCount, index]);

  return (
    <Button colorScheme="base" pointerEvents="none" minW={28}>
      {counterText}
    </Button>
  );
});

StagingAreaToolbarImageCountButton.displayName = 'StagingAreaToolbarImageCountButton';
