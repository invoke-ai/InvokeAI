import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectGalleryLayoutMode } from 'features/gallery/store/gallerySelectors';
import { galleryLayoutModeChanged } from 'features/gallery/store/gallerySlice';
import { selectShouldUsePagedGalleryView } from 'features/ui/store/uiSelectors';
import { setShouldUsePagedGalleryView } from 'features/ui/store/uiSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiColumnsFill, PiGridFourFill } from 'react-icons/pi';

import { getEffectiveGalleryLayout } from './galleryLayout';

type GalleryLayoutToggleProps = {
  isDisabled?: boolean;
};

export const GalleryLayoutToggle = memo(({ isDisabled = false }: GalleryLayoutToggleProps) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const galleryLayoutMode = useAppSelector(selectGalleryLayoutMode);
  const shouldUsePagedGalleryView = useAppSelector(selectShouldUsePagedGalleryView);
  const effectiveGalleryLayout = getEffectiveGalleryLayout(galleryLayoutMode, shouldUsePagedGalleryView);

  const handleClickGrid = useCallback(() => {
    dispatch(galleryLayoutModeChanged('grid'));
  }, [dispatch]);

  const handleClickMasonry = useCallback(() => {
    dispatch(galleryLayoutModeChanged('masonry'));
    dispatch(setShouldUsePagedGalleryView(false));
  }, [dispatch]);

  return (
    <ButtonGroup size="sm" variant="outline" isAttached flexShrink={0}>
      <IconButton
        aria-label={t('gallery.galleryLayoutGrid')}
        colorScheme={effectiveGalleryLayout === 'grid' ? 'invokeBlue' : undefined}
        data-testid="gallery-layout-grid-button"
        icon={<PiGridFourFill />}
        isDisabled={isDisabled}
        onClick={handleClickGrid}
        tooltip={t('gallery.galleryLayoutGridTooltip')}
      />
      <IconButton
        aria-label={t('gallery.galleryLayoutMasonry')}
        colorScheme={effectiveGalleryLayout === 'masonry' ? 'invokeBlue' : undefined}
        data-testid="gallery-layout-masonry-button"
        icon={<PiColumnsFill />}
        isDisabled={isDisabled}
        onClick={handleClickMasonry}
        tooltip={t('gallery.galleryLayoutMasonryTooltip')}
      />
    </ButtonGroup>
  );
});

GalleryLayoutToggle.displayName = 'GalleryLayoutToggle';
