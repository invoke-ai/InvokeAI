import { Button, ButtonGroup } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { comparisonModeChanged } from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';

export const ImageComparisonToolbarButtons = memo(() => {
  const dispatch = useAppDispatch();
  const comparisonMode = useAppSelector((s) => s.gallery.comparisonMode);
  const setComparisonModeSlider = useCallback(() => {
    dispatch(comparisonModeChanged('slider'));
  }, [dispatch]);
  const setComparisonModeSideBySide = useCallback(() => {
    dispatch(comparisonModeChanged('side-by-side'));
  }, [dispatch]);
  const setComparisonModeOverlay = useCallback(() => {
    dispatch(comparisonModeChanged('overlay'));
  }, [dispatch]);

  return (
    <>
      <ButtonGroup variant="outline">
        <Button onClick={setComparisonModeSlider} colorScheme={comparisonMode === 'slider' ? 'invokeBlue' : 'base'}>
          Slider
        </Button>
        <Button
          onClick={setComparisonModeSideBySide}
          colorScheme={comparisonMode === 'side-by-side' ? 'invokeBlue' : 'base'}
        >
          Side-by-Side
        </Button>
        <Button onClick={setComparisonModeOverlay} colorScheme={comparisonMode === 'overlay' ? 'invokeBlue' : 'base'}>
          Overlay
        </Button>
      </ButtonGroup>
    </>
  );
});

ImageComparisonToolbarButtons.displayName = 'ImageComparisonToolbarButtons';
