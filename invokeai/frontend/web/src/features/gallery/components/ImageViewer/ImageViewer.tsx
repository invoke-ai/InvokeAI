import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useScopeOnFocus, useScopeOnMount } from 'common/hooks/interactionScopes';
import { CompareToolbar } from 'features/gallery/components/ImageViewer/CompareToolbar';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { ImageComparisonDroppable } from 'features/gallery/components/ImageViewer/ImageComparisonDroppable';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { memo, useRef } from 'react';
import { useMeasure } from 'react-use';

export const ImageViewer = memo(() => {
  const isComparing = useAppSelector((s) => s.gallery.imageToCompare !== null);
  const [containerRef, containerDims] = useMeasure<HTMLDivElement>();
  const ref = useRef<HTMLDivElement>(null);
  useScopeOnFocus('imageViewer', ref);
  useScopeOnMount('imageViewer');

  return (
    <Flex
      ref={ref}
      tabIndex={-1}
      layerStyle="first"
      borderRadius="base"
      position="absolute"
      flexDirection="column"
      top={0}
      right={0}
      bottom={0}
      left={0}
      p={2}
      rowGap={4}
      alignItems="center"
      justifyContent="center"
    >
      {isComparing && <CompareToolbar />}
      {!isComparing && <ViewerToolbar />}
      <Box ref={containerRef} w="full" h="full">
        {!isComparing && <CurrentImagePreview />}
        {isComparing && <ImageComparison containerDims={containerDims} />}
      </Box>
      <ImageComparisonDroppable />
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';
