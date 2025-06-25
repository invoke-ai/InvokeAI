import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { clamp } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { useGalleryImageNames } from './use-gallery-image-names';

const NextPrevImageButtons = ({ inset = 8 }: { inset?: ChakraProps['insetInlineStart' | 'insetInlineEnd'] }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const { imageNames, isFetching } = useGalleryImageNames();

  const isOnFirstImage = useMemo(
    () => (lastSelectedImage ? imageNames.at(0) === lastSelectedImage : false),
    [imageNames, lastSelectedImage]
  );
  const isOnLastImage = useMemo(
    () => (lastSelectedImage ? imageNames.at(-1) === lastSelectedImage : false),
    [imageNames, lastSelectedImage]
  );

  const onClickLeftArrow = useCallback(() => {
    const targetIndex = lastSelectedImage ? imageNames.findIndex((n) => n === lastSelectedImage) - 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, lastSelectedImage]);

  const onClickRightArrow = useCallback(() => {
    const targetIndex = lastSelectedImage ? imageNames.findIndex((n) => n === lastSelectedImage) + 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, lastSelectedImage]);

  return (
    <Box pos="relative" h="full" w="full">
      {!isOnFirstImage && (
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
      {!isOnLastImage && (
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
