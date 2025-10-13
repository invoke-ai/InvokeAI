import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clamp } from 'es-toolkit/compat';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { itemSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';

import { useGalleryImageNames } from './use-gallery-image-names';

const NextPrevItemButtons = ({ inset = 8 }: { inset?: ChakraProps['insetInlineStart' | 'insetInlineEnd'] }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const { imageNames, isFetching } = useGalleryImageNames();

  const isOnFirstItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(0) === lastSelectedItem.id : false),
    [imageNames, lastSelectedItem]
  );
  const isOnLastItem = useMemo(
    () => (lastSelectedItem ? imageNames.at(-1) === lastSelectedItem.id : false),
    [imageNames, lastSelectedItem]
  );

  const onClickLeftArrow = useCallback(() => {
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem.id) - 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(itemSelected({ type: lastSelectedItem?.type ?? 'image', id: n }));
  }, [dispatch, imageNames, lastSelectedItem]);

  const onClickRightArrow = useCallback(() => {
    const targetIndex = lastSelectedItem ? imageNames.findIndex((n) => n === lastSelectedItem.id) + 1 : 0;
    const clampedIndex = clamp(targetIndex, 0, imageNames.length - 1);
    const n = imageNames.at(clampedIndex);
    if (!n) {
      return;
    }
    dispatch(itemSelected({ type: lastSelectedItem?.type ?? 'image', id: n }));
  }, [dispatch, imageNames, lastSelectedItem]);

  return (
    <Box pos="relative" h="full" w="full">
      {!isOnFirstItem && (
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
      {!isOnLastItem && (
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

export default memo(NextPrevItemButtons);
