import { Box, Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { debounce } from 'es-toolkit';
import type { ComparisonWrapperProps } from 'features/gallery/components/ImageViewer/common';
import { selectImageToCompare } from 'features/gallery/components/ImageViewer/common';
import { CompareToolbar } from 'features/gallery/components/ImageViewer/CompareToolbar';
import { ImageComparisonDroppable } from 'features/gallery/components/ImageViewer/ImageComparisonDroppable';
import { ImageComparisonHover } from 'features/gallery/components/ImageViewer/ImageComparisonHover';
import { ImageComparisonSideBySide } from 'features/gallery/components/ImageViewer/ImageComparisonSideBySide';
import { ImageComparisonSlider } from 'features/gallery/components/ImageViewer/ImageComparisonSlider';
import { selectComparisonMode, selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useLayoutEffect, useRef, useState } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const ImageComparisonContent = memo(({ firstImage, secondImage, rect }: ComparisonWrapperProps) => {
  const comparisonMode = useAppSelector(selectComparisonMode);

  if (!firstImage || !secondImage) {
    return null;
  }

  if (comparisonMode === 'slider') {
    return <ImageComparisonSlider firstImage={firstImage} secondImage={secondImage} rect={rect} />;
  }

  if (comparisonMode === 'side-by-side') {
    return <ImageComparisonSideBySide firstImage={firstImage} secondImage={secondImage} rect={rect} />;
  }

  if (comparisonMode === 'hover') {
    return <ImageComparisonHover firstImage={firstImage} secondImage={secondImage} rect={rect} />;
  }

  assert<Equals<never, typeof comparisonMode>>(false);
});

ImageComparisonContent.displayName = 'ImageComparisonContent';

export const ImageComparison = memo(() => {
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const lastSelectedImageDTO = useImageDTO(lastSelectedItem);
  const comparisonImageDTO = useImageDTO(useAppSelector(selectImageToCompare));

  const [rect, setRect] = useState<DOMRect | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  // Ref callback runs synchronously when the DOM node is attached, ensuring we have a measurement before
  // the comparison content is rendered.
  const measureNode = useCallback((node: HTMLDivElement) => {
    if (node) {
      ref.current = node;
      const boundingRect = node.getBoundingClientRect();
      setRect(boundingRect);
    }
  }, []);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    const measureRect = debounce(() => {
      const boundingRect = el.getBoundingClientRect();
      setRect(boundingRect);
    }, 300);
    const observer = new ResizeObserver(measureRect);
    observer.observe(el);
    return () => {
      observer.disconnect();
    };
  }, []);

  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <CompareToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative">
        <Box ref={measureNode} w="full" h="full" overflow="hidden">
          <ImageComparisonContent firstImage={lastSelectedImageDTO} secondImage={comparisonImageDTO} rect={rect} />
        </Box>
        <ImageComparisonDroppable />
      </Flex>
    </Flex>
  );
});
ImageComparison.displayName = 'ImageComparison';
