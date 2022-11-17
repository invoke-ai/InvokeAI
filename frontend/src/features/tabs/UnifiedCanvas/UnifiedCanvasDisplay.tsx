import { createSelector } from '@reduxjs/toolkit';
// import IAICanvas from 'features/canvas/IAICanvas';
import IAICanvasResizer from 'features/canvas/IAICanvasResizer';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { useAppDispatch, useAppSelector } from 'app/store';
import ImageUploadButton from 'common/components/ImageUploaderButton';
import {
  canvasSelector,
  setDoesCanvasNeedScaling,
} from 'features/canvas/canvasSlice';
import IAICanvas from 'features/canvas/IAICanvas';
import IAICanvasOutpaintingControls from 'features/canvas/IAICanvasOutpaintingControls';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      doesCanvasNeedScaling,
      layerState: { objects },
    } = canvas;
    return {
      doesCanvasNeedScaling,
      doesOutpaintingHaveObjects: objects.length > 0,
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
  const { doesCanvasNeedScaling, doesOutpaintingHaveObjects } =
    useAppSelector(selector);

  useLayoutEffect(() => {
    const resizeCallback = _.debounce(
      () => dispatch(setDoesCanvasNeedScaling(true)),
      250
    );
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
