import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clamp } from 'es-toolkit/compat';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { useGalleryImageNames } from './use-gallery-image-names';

const ARROW_SIZE = 48;

const NextPrevItemButtons = ({ inset = 8 }: { inset?: ChakraProps['insetInlineStart' | 'insetInlineEnd'] }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const { imageNames, isFetching } = useGalleryImageNames();

  const isOnFirstItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(0) === lastSelectedItem : false),
    [imageNames, lastSelectedItem]
  );
  const isOnLastItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(-1) === lastSelectedItem : false),
    [imageNames, lastSelectedItem]
  );

  const onClickLeftArrow = useCallback(() => {
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) - 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, lastSelectedItem]);

  const onClickRightArrow = useCallback(() => {
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem) + 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(imageSelected(n));
  }, [dispatch, imageNames, lastSelectedItem]);

  return (
    <Box pos="relative" h="full" w="full">
      {!isOnFirstItem && (
        <IconButton
          position="absolute"
          top="50%"
          transform="translate(0, -50%)"
          aria-label={t('accessibility.previousImage')}
          icon={<PiCaretLeftBold size={ARROW_SIZE} />}
          variant="unstyled"
          padding={0}
          minW={0}
          minH={0}
          w={`${ARROW_SIZE}px`}
          h={`${ARROW_SIZE}px`}
          onClick={onClickLeftArrow}
          isDisabled={isFetching}
          color="base.100"
          pointerEvents="auto"
          insetInlineStart={inset}
        />
      )}
      {!isOnLastItem && (
        <IconButton
          position="absolute"
          top="50%"
          transform="translate(0, -50%)"
          aria-label={t('accessibility.nextImage')}
          icon={<PiCaretRightBold size={ARROW_SIZE} />}
          variant="unstyled"
          padding={0}
          minW={0}
          minH={0}
          w={`${ARROW_SIZE}px`}
          h={`${ARROW_SIZE}px`}
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

export default memo(NextPrevItemButtons);
