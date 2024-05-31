import { Button, ButtonGroup } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { comparedImagesSwapped, comparisonModeChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ImageComparisonToolbarButtons = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const comparisonMode = useAppSelector((s) => s.gallery.comparisonMode);
  const setComparisonModeSlider = useCallback(() => {
    dispatch(comparisonModeChanged('slider'));
  }, [dispatch]);
  const setComparisonModeSideBySide = useCallback(() => {
    dispatch(comparisonModeChanged('side-by-side'));
  }, [dispatch]);
  const swapImages = useCallback(() => {
    dispatch(comparedImagesSwapped());
  }, [dispatch]);

  return (
    <>
      <ButtonGroup variant="outline">
        <Button onClick={setComparisonModeSlider} colorScheme={comparisonMode === 'slider' ? 'invokeBlue' : 'base'}>
          {t('gallery.slider')}
        </Button>
        <Button
          onClick={setComparisonModeSideBySide}
          colorScheme={comparisonMode === 'side-by-side' ? 'invokeBlue' : 'base'}
        >
          {t('gallery.sideBySide')}
        </Button>
      </ButtonGroup>
      <Button onClick={swapImages}>{t('gallery.swapImages')}</Button>
    </>
  );
});

ImageComparisonToolbarButtons.displayName = 'ImageComparisonToolbarButtons';
