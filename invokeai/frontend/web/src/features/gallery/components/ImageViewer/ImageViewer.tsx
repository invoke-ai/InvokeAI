import { Box, Flex } from '@invoke-ai/ui-library';
import { useMeasure } from '@reactuses/core';
import { useAppSelector } from 'app/store/storeHooks';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageSliderComparison } from 'features/gallery/components/ImageViewer/ImageSliderComparison';
import { ToggleMetadataViewerButton } from 'features/gallery/components/ImageViewer/ToggleMetadataViewerButton';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

import CurrentImageButtons from './CurrentImageButtons';
import { ViewerToggleMenu } from './ViewerToggleMenu';

const VIEWER_ENABLED_TABS: InvokeTabName[] = ['canvas', 'generation', 'workflows'];

export const ImageViewer = memo(() => {
  const { isOpen, onToggle, onClose } = useImageViewer();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const isViewerEnabled = useMemo(() => VIEWER_ENABLED_TABS.includes(activeTabName), [activeTabName]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize] = useMeasure(containerRef);
  const shouldShowViewer = useMemo(() => {
    if (!isViewerEnabled) {
      return false;
    }
    return isOpen;
  }, [isOpen, isViewerEnabled]);

  useHotkeys('z', onToggle, { enabled: isViewerEnabled }, [isViewerEnabled, onToggle]);
  useHotkeys('esc', onClose, { enabled: isViewerEnabled }, [isViewerEnabled, onClose]);

  const { firstImage, secondImage } = useAppSelector((s) => {
    const images = s.gallery.selection.slice(-2);
    return { firstImage: images[0] ?? null, secondImage: images[0] ? images[1] ?? null : null };
  });

  if (!shouldShowViewer) {
    return null;
  }

  return (
    <Flex
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
      zIndex={10} // reactflow puts its minimap at 5, so we need to be above that
    >
      <Flex w="full" gap={2}>
        <Flex flex={1} justifyContent="center">
          <Flex gap={2} marginInlineEnd="auto">
            <ToggleProgressButton />
            <ToggleMetadataViewerButton />
          </Flex>
        </Flex>
        <Flex flex={1} gap={2} justifyContent="center">
          <CurrentImageButtons />
        </Flex>
        <Flex flex={1} justifyContent="center">
          <Flex gap={2} marginInlineStart="auto">
            <ViewerToggleMenu />
          </Flex>
        </Flex>
      </Flex>
      <Box ref={containerRef} w="full" h="full">
        {firstImage && !secondImage && <CurrentImagePreview />}
        {firstImage && secondImage && (
          <ImageSliderComparison containerSize={containerSize} firstImage={firstImage} secondImage={secondImage} />
        )}
      </Box>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';
