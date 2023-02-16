import { createSelector } from '@reduxjs/toolkit';
// import IAICanvas from 'features/canvas/components/IAICanvas';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import IAICanvasOutpaintingControls from 'features/canvas/components/IAICanvasToolbar/IAICanvasToolbar';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { debounce, isEqual } from 'lodash';

import { useLayoutEffect } from 'react';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { doesCanvasNeedScaling } = canvas;
    return {
      doesCanvasNeedScaling,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const UnifiedCanvasDisplay = () => {
  const dispatch = useAppDispatch();

  const { doesCanvasNeedScaling } = useAppSelector(selector);

  useLayoutEffect(() => {
    dispatch(setDoesCanvasNeedScaling(true));

    const resizeCallback = debounce(() => {
      dispatch(setDoesCanvasNeedScaling(true));
    }, 250);

    window.addEventListener('resize', resizeCallback);

    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  return (
    <div className="workarea-single-view">
      <div className="workarea-split-view-left">
        <div className="inpainting-main-area">
          <IAICanvasOutpaintingControls />
          <div className="inpainting-canvas-area">
            {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
          </div>
        </div>
      </div>
    </div>
  );
};

export default UnifiedCanvasDisplay;
