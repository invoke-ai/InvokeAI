import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, IconButton } from '@invoke-ai/ui-library';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { useGalleryNavigation } from 'features/gallery/hooks/useGalleryNavigation';
import { useGalleryPagination } from 'features/gallery/hooks/useGalleryPagination';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

const NextPrevImageButtons = ({ inset = 8 }: { inset?: ChakraProps['insetInlineStart' | 'insetInlineEnd'] }) => {
  const { t } = useTranslation();
  const { prevImage, nextImage, isOnFirstImageOfView, isOnLastImageOfView } = useGalleryNavigation();

  const { isFetching } = useGalleryImages().queryResult;
  const { isNextEnabled, goNext, isPrevEnabled, goPrev } = useGalleryPagination();

  const shouldShowLeftArrow = useMemo(() => {
    if (!isOnFirstImageOfView) {
      return true;
    }
    if (isPrevEnabled) {
      return true;
    }
    return false;
  }, [isOnFirstImageOfView, isPrevEnabled]);

  const onClickLeftArrow = useCallback(() => {
    if (isOnFirstImageOfView) {
      if (isPrevEnabled && !isFetching) {
        goPrev('arrow');
      }
    } else {
      prevImage();
    }
  }, [goPrev, isFetching, isOnFirstImageOfView, isPrevEnabled, prevImage]);

  const shouldShowRightArrow = useMemo(() => {
    if (!isOnLastImageOfView) {
      return true;
    }
    if (isNextEnabled) {
      return true;
    }
    return false;
  }, [isNextEnabled, isOnLastImageOfView]);

  const onClickRightArrow = useCallback(() => {
    if (isOnLastImageOfView) {
      if (isNextEnabled && !isFetching) {
        goNext('arrow');
      }
    } else {
      nextImage();
    }
  }, [goNext, isFetching, isNextEnabled, isOnLastImageOfView, nextImage]);

  return (
    <Box pos="relative" h="full" w="full">
      {shouldShowLeftArrow && (
        <IconButton
          position="absolute"
          top="50%"
          transform="translate(0, -50%)"
          aria-label={t('accessibility.previousImage')}
          icon={<PiCaretLeftBold size={64} />}
          variant="unstyled"
          onClick={onClickLeftArrow}
          isDisabled={isFetching}
          color="base.100"
          pointerEvents="auto"
          insetInlineStart={inset}
        />
      )}
      {shouldShowRightArrow && (
        <IconButton
          position="absolute"
          top="50%"
          transform="translate(0, -50%)"
          aria-label={t('accessibility.nextImage')}
          icon={<PiCaretRightBold size={64} />}
          variant="unstyled"
          onClick={onClickRightArrow}
          isDisabled={isFetching}
          color="base.100"
          pointerEvents="auto"
          insetInlineEnd={inset}
        />
      )}
    </Box>
  );
};

export default memo(NextPrevImageButtons);
