import { Tag, TagCloseButton, TagLabel } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { FOCUS_REGIONS } from 'common/hooks/interactionScopes';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const GallerySelectionCountTag = () => {
  const dispatch = useAppDispatch();
  const { selection } = useAppSelector((s) => s.gallery);
  const { t } = useTranslation();
  const { imageDTOs } = useGalleryImages();
  const isFocusedOnGallery = useStore(FOCUS_REGIONS.galleryPanel.$isFocused);

  const onClearSelection = useCallback(() => {
    const firstImage = selection[0];
    if (firstImage) {
      dispatch(selectionChanged([firstImage]));
    }
  }, [dispatch, selection]);

  const onSelectPage = useCallback(() => {
    dispatch(selectionChanged([...selection, ...imageDTOs]));
  }, [dispatch, selection, imageDTOs]);

  useRegisteredHotkeys({
    id: 'selectAllOnPage',
    category: 'gallery',
    callback: onSelectPage,
    options: { preventDefault: true, enabled: isFocusedOnGallery },
    dependencies: [onSelectPage, isFocusedOnGallery],
  });

  useRegisteredHotkeys({
    id: 'clearSelection',
    category: 'gallery',
    callback: onClearSelection,
    options: { enabled: selection.length > 0 && isFocusedOnGallery },
    dependencies: [onClearSelection, selection, isFocusedOnGallery],
  });

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
