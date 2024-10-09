import { Box, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion } from 'common/hooks/focus';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { CanvasAlertsSendingToCanvas } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSendingTo';
import { CompareToolbar } from 'features/gallery/components/ImageViewer/CompareToolbar';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { ImageComparisonDroppable } from 'features/gallery/components/ImageViewer/ImageComparisonDroppable';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { selectHasImageToCompare } from 'features/gallery/store/gallerySelectors';
import type { ReactNode } from 'react';
import { memo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useMeasure } from 'react-use';

import { useImageViewer } from './useImageViewer';

type Props = {
  closeButton?: ReactNode;
};

const useFocusRegionOptions = {
  focusOnMount: true,
};

export const ImageViewer = memo(({ closeButton }: Props) => {
  useAssertSingleton('ImageViewer');
  const hasImageToCompare = useAppSelector(selectHasImageToCompare);
  const [containerRef, containerDims] = useMeasure<HTMLDivElement>();
  const ref = useRef<HTMLDivElement>(null);
  useFocusRegion('viewer', ref, useFocusRegionOptions);

  return (
    <Flex
      ref={ref}
      tabIndex={-1}
      layerStyle="first"
      p={2}
      borderRadius="base"
      position="absolute"
      flexDirection="column"
      top={0}
      right={0}
      bottom={0}
      left={0}
      rowGap={4}
      alignItems="center"
      justifyContent="center"
    >
      {hasImageToCompare && <CompareToolbar />}
      {!hasImageToCompare && <ViewerToolbar closeButton={closeButton} />}
      <Box ref={containerRef} w="full" h="full">
        {!hasImageToCompare && <CurrentImagePreview />}
        {hasImageToCompare && <ImageComparison containerDims={containerDims} />}
      </Box>
      <ImageComparisonDroppable />
      <Box position="absolute" top={14} insetInlineEnd={2}>
        <CanvasAlertsSendingToCanvas />
      </Box>
    </Flex>
  );
});

ImageViewer.displayName = 'ImageViewer';

export const GatedImageViewer = memo(() => {
  const imageViewer = useImageViewer();

  if (!imageViewer.isOpen) {
    return null;
  }

  return <ImageViewer closeButton={<ImageViewerCloseButton />} />;
});

GatedImageViewer.displayName = 'GatedImageViewer';

const ImageViewerCloseButton = memo(() => {
  const { t } = useTranslation();
  const imageViewer = useImageViewer();
  useAssertSingleton('ImageViewerCloseButton');
  useHotkeys('esc', imageViewer.close);
  return (
    <IconButton
      tooltip={t('gallery.closeViewer')}
      aria-label={t('gallery.closeViewer')}
      icon={<PiXBold />}
      variant="ghost"
      onClick={imageViewer.close}
    />
  );
});

ImageViewerCloseButton.displayName = 'ImageViewerCloseButton';

const GatedImageViewerCloseButton = memo(() => {
  const imageViewer = useImageViewer();

  if (!imageViewer.isOpen) {
    return null;
  }

  return <ImageViewerCloseButton />;
});

GatedImageViewerCloseButton.displayName = 'GatedImageViewerCloseButton';
