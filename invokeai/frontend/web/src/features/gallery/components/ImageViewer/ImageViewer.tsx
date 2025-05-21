import { Box, Flex, IconButton, type SystemStyleObject, useOutsideClick } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectImageToCompare } from 'features/gallery/components/ImageViewer/common';
import CurrentImagePreview from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar';
import { selectLastSelectedImageName } from 'features/gallery/store/gallerySelectors';
import type { ReactNode } from 'react';
import { memo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

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

export const ImageViewer = memo(() => {
  const lastSelectedImageName = useAppSelector(selectLastSelectedImageName);
  const { data: lastSelectedImageDTO } = useGetImageDTOQuery(lastSelectedImageName ?? skipToken);
  const comparisonImageDTO = useAppSelector(selectImageToCompare);

  if (lastSelectedImageDTO && comparisonImageDTO) {
    return <ImageComparison firstImage={lastSelectedImageDTO} secondImage={comparisonImageDTO} />;
  }

  return <CurrentImagePreview imageDTO={lastSelectedImageDTO} />;
});

ImageViewer.displayName = 'ImageViewer';

const imageViewerContainerSx: SystemStyleObject = {
  position: 'absolute',
  top: 0,
  right: 0,
  bottom: 0,
  left: 0,
  transition: 'opacity 0.15s ease',
  opacity: 1,
  pointerEvents: 'auto',
  '&[data-hidden="true"]': {
    opacity: 0,
    pointerEvents: 'none',
  },
  backdropFilter: 'blur(10px) brightness(70%)',
};

export const ImageViewerModal = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  const imageViewer = useImageViewer();
  useOutsideClick({
    ref,
    handler: imageViewer.close,
  });

  return (
    <Box sx={imageViewerContainerSx} data-hidden={!imageViewer.isOpen}>
      <Flex
        ref={ref}
        flexDir="column"
        position="absolute"
        bg="base.900"
        borderRadius="base"
        top={16}
        right={16}
        bottom={16}
        left={16}
      >
        <ViewerToolbar />
        <ImageViewer />
      </Flex>
    </Box>
  );
});

ImageViewerModal.displayName = 'GatedImageViewer';

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
