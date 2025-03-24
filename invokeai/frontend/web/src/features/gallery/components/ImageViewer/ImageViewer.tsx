import { Box, IconButton, type SystemStyleObject } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { CompareToolbar } from 'features/gallery/components/ImageViewer/CompareToolbar';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { ImageComparisonDroppable } from 'features/gallery/components/ImageViewer/ImageComparisonDroppable';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { selectHasImageToCompare } from 'features/gallery/store/gallerySelectors';
import type { ReactNode } from 'react';
import { memo } from 'react';
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

const FOCUS_REGION_STYLES: SystemStyleObject = {
  display: 'flex',
  width: 'full',
  height: 'full',
  position: 'absolute',
  flexDirection: 'column',
  inset: 0,
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
};

export const ImageViewer = memo(({ closeButton }: Props) => {
  useAssertSingleton('ImageViewer');
  const hasImageToCompare = useAppSelector(selectHasImageToCompare);
  const [containerRef, containerDims] = useMeasure<HTMLDivElement>();

  return (
    <FocusRegionWrapper region="viewer" sx={FOCUS_REGION_STYLES} layerStyle="first" {...useFocusRegionOptions}>
      {hasImageToCompare && <CompareToolbar />}
      {!hasImageToCompare && <ViewerToolbar closeButton={closeButton} />}
      <Box ref={containerRef} w="full" h="full" p={2} overflow="hidden">
        {!hasImageToCompare && <CurrentImagePreview />}
        {hasImageToCompare && <ImageComparison containerDims={containerDims} />}
      </Box>
      <ImageComparisonDroppable />
    </FocusRegionWrapper>
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
      variant="link"
      alignSelf="stretch"
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
