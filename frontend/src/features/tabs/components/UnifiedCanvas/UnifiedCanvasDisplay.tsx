import { createSelector } from '@reduxjs/toolkit';
// import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { useAppDispatch, useAppSelector } from 'app/store';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasOutpaintingControls from 'features/canvas/components/IAICanvasToolbar/IAICanvasToolbar';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

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
      resultEqualityCheck: _.isEqual,
    },
  }
);

const UnifiedCanvasDisplay = () => {
  const dispatch = useAppDispatch();

  const { doesCanvasNeedScaling } = useAppSelector(selector);

  useLayoutEffect(() => {
    dispatch(setDoesCanvasNeedScaling(true));

    const resizeCallback = _.debounce(() => {
      dispatch(setDoesCanvasNeedScaling(true));
    }, 250);

    window.addEventListener('resize', resizeCallback);

    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  return (
    <div className={'workarea-single-view'}>
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
