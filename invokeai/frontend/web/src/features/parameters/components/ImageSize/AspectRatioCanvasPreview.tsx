import { useStore } from '@nanostores/react';
import { $isPreviewVisible } from 'features/controlLayers/store/canvasV2Slice';
import { AspectRatioIconPreview } from 'features/parameters/components/ImageSize/AspectRatioIconPreview';
import { memo } from 'react';

export const AspectRatioCanvasPreview = memo(() => {
  const isPreviewVisible = useStore($isPreviewVisible);

  return <AspectRatioIconPreview />;
  // if (!isPreviewVisible) {
  //   return <AspectRatioIconPreview />;
  // }

  // return (
  //   <Flex w="full" h="full" alignItems="center" justifyContent="center" position="relative">
  //     <StageComponent asPreview />
  //   </Flex>
  // );
});

AspectRatioCanvasPreview.displayName = 'AspectRatioCanvasPreview';
