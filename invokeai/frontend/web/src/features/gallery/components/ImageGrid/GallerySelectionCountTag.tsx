import { Tag, TagCloseButton, TagLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { selectionChanged } from 'features/gallery/store/gallerySlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { ImageDTO } from 'services/api/types';

export const GallerySelectionCountTag = memo(() => {
  const dispatch = useAppDispatch();
  const { selection } = useAppSelector((s) => s.gallery);
  const { imageDTOs } = useGalleryImages();
  const isGalleryFocused = useIsRegionFocused('gallery');

  const onSelectPage = useCallback(() => {
    dispatch(selectionChanged([...selection, ...imageDTOs]));
  }, [dispatch, selection, imageDTOs]);

  useRegisteredHotkeys({
    id: 'selectAllOnPage',
    category: 'gallery',
    callback: onSelectPage,
    options: { preventDefault: true, enabled: isGalleryFocused },
    dependencies: [onSelectPage, isGalleryFocused],
  });

  if (selection.length <= 1) {
    return null;
  }

  return <GallerySelectionCountTagContent selection={selection} />;
});

GallerySelectionCountTag.displayName = 'GallerySelectionCountTag';

const GallerySelectionCountTagContent = memo(({ selection }: { selection: ImageDTO[] }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const isGalleryFocused = useIsRegionFocused('gallery');

  const onClearSelection = useCallback(() => {
    const firstImage = selection[0];
    if (firstImage) {
      dispatch(selectionChanged([firstImage]));
    }
  }, [dispatch, selection]);

  useRegisteredHotkeys({
    id: 'clearSelection',
    category: 'gallery',
    callback: onClearSelection,
    options: { enabled: selection.length > 0 && isGalleryFocused },
    dependencies: [onClearSelection, selection, isGalleryFocused],
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
        {selection.length} {t('common.selected')}
      </TagLabel>
      <TagCloseButton onClick={onClearSelection} />
    </Tag>
  );
});

GallerySelectionCountTagContent.displayName = 'GallerySelectionCountTagContent';
