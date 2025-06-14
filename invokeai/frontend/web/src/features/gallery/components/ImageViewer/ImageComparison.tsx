import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { ComparisonProps } from 'features/gallery/components/ImageViewer/common';
import { CompareToolbar } from 'features/gallery/components/ImageViewer/CompareToolbar';
import { ImageComparisonDroppable } from 'features/gallery/components/ImageViewer/ImageComparisonDroppable';
import { ImageComparisonHover } from 'features/gallery/components/ImageViewer/ImageComparisonHover';
import { ImageComparisonSideBySide } from 'features/gallery/components/ImageViewer/ImageComparisonSideBySide';
import { ImageComparisonSlider } from 'features/gallery/components/ImageViewer/ImageComparisonSlider';
import { selectComparisonMode } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useMeasure } from 'react-use';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

export const ImageComparisonContent = memo(({ firstImage, secondImage, containerDims }: ComparisonProps) => {
  const comparisonMode = useAppSelector(selectComparisonMode);

  if (comparisonMode === 'slider') {
    return <ImageComparisonSlider firstImage={firstImage} secondImage={secondImage} containerDims={containerDims} />;
  }

  if (comparisonMode === 'side-by-side') {
    return (
      <ImageComparisonSideBySide firstImage={firstImage} secondImage={secondImage} containerDims={containerDims} />
    );
  }

  if (comparisonMode === 'hover') {
    return <ImageComparisonHover firstImage={firstImage} secondImage={secondImage} containerDims={containerDims} />;
  }

  assert<Equals<never, typeof comparisonMode>>(false);
});

ImageComparisonContent.displayName = 'ImageComparisonContent';

export const ImageComparison = memo(({ firstImage, secondImage }: Omit<ComparisonProps, 'containerDims'>) => {
  const [containerRef, containerDims] = useMeasure<HTMLDivElement>();

  return (
    <Flex flexDir="column" w="full" h="full" position="relative">
      <CompareToolbar />
      <Box ref={containerRef} w="full" h="full" p={2} overflow="hidden">
        <ImageComparisonContent firstImage={firstImage} secondImage={secondImage} containerDims={containerDims} />
      </Box>
      <ImageComparisonDroppable />
    </Flex>
  );
});
ImageComparison.displayName = 'ImageComparison';
