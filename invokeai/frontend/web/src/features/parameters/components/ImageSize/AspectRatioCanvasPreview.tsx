import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { StageComponent } from 'features/controlLayers/components/StageComponent';
import { $isPreviewVisible } from 'features/controlLayers/store/canvasV2Slice';
import { memo } from 'react';

export const AspectRatioCanvasPreview = memo(() => {
  const isPreviewVisible = useStore($isPreviewVisible);

  // if (!isPreviewVisible) {
  //   return <AspectRatioIconPreview />;
  // }

  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center" position="relative">
      <StageComponent asPreview />
    </Flex>
  );
});

AspectRatioCanvasPreview.displayName = 'AspectRatioCanvasPreview';
