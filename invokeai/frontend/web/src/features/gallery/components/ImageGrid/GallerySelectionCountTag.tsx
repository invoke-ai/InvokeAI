import { Tag, TagCloseButton, TagLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const GallerySelectionCountTag = () => {
  const dispatch = useAppDispatch();
  const { selection } = useAppSelector((s) => s.gallery);
  const { t } = useTranslation();
  const { imageDTOs } = useGalleryImages();

  const onClearSelection = useCallback(() => {
    dispatch(selectionChanged([]));
  }, [dispatch]);

  const onSelectPage = useCallback(() => {
    dispatch(selectionChanged(imageDTOs));
  }, [dispatch, imageDTOs]);

  useHotkeys(['ctrl+a', 'meta+a'], onSelectPage, { preventDefault: true }, [onSelectPage]);

  if (selection.length <= 1) {
    return null;
  }

  return (
    <Tag
      position="absolute"
      bg="invokeBlue.800"
      color="base.50"
      py={2}
      px={4}
      userSelect="none"
      shadow="dark-lg"
      fontWeight="semibold"
      border={1}
      borderStyle="solid"
      borderColor="whiteAlpha.300"
    >
      <TagLabel>
        {selection.length} {t('common.selected')}
      </TagLabel>
      <TagCloseButton onClick={onClearSelection} />
    </Tag>
  );
};
