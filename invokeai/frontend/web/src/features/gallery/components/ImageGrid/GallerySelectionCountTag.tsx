import { Tag, TagCloseButton, TagLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector, useAppStore } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useGalleryImageNames } from 'features/gallery/components/use-gallery-image-names';
import { selectFirstSelectedImage, selectSelectionCount } from 'features/gallery/store/gallerySelectors';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const GallerySelectionCountTag = memo(() => {
  const { dispatch } = useAppStore();
  const selectionCount = useAppSelector(selectSelectionCount);
  const { imageNames } = useGalleryImageNames();
  const isGalleryFocused = useIsRegionFocused('gallery');

  const onSelectPage = useCallback(() => {
    dispatch(selectionChanged([...imageNames]));
  }, [dispatch, imageNames]);

  useRegisteredHotkeys({
    id: 'selectAllOnPage',
    category: 'gallery',
    callback: onSelectPage,
    options: { preventDefault: true, enabled: isGalleryFocused },
    dependencies: [onSelectPage, isGalleryFocused],
  });

  if (selectionCount <= 1) {
    return null;
  }

  return <GallerySelectionCountTagContent />;
});

GallerySelectionCountTag.displayName = 'GallerySelectionCountTag';

const GallerySelectionCountTagContent = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isGalleryFocused = useIsRegionFocused('gallery');
  const firstImage = useAppSelector(selectFirstSelectedImage);
  const selectionCount = useAppSelector(selectSelectionCount);
  const onClearSelection = useCallback(() => {
    if (firstImage) {
      dispatch(selectionChanged([firstImage]));
    }
  }, [dispatch, firstImage]);

  useRegisteredHotkeys({
    id: 'clearSelection',
    category: 'gallery',
    callback: onClearSelection,
    options: { enabled: selectionCount > 0 && isGalleryFocused },
    dependencies: [onClearSelection, selectionCount, isGalleryFocused],
  });

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
        {selectionCount} {t('common.selected')}
      </TagLabel>
      <TagCloseButton onClick={onClearSelection} />
    </Tag>
  );
});

GallerySelectionCountTagContent.displayName = 'GallerySelectionCountTagContent';
