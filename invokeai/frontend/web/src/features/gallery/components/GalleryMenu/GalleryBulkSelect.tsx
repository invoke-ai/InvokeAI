import { Spacer, Tag, TagCloseButton, TagLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

export const GalleryBulkSelect = () => {
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
    return <Spacer />;
  }

  return (
    <Tag py={1} px={3} userSelect="none" border={1} borderStyle="solid" borderColor="whiteAlpha.300">
      <TagLabel>
        {selection.length} {t('common.selected')}
      </TagLabel>
      <TagCloseButton onClick={onClearSelection} />
    </Tag>
  );
};
