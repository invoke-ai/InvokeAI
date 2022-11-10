import { createSelector } from '@reduxjs/toolkit';
// import IAICanvas from 'features/canvas/IAICanvas';
import IAICanvasControls from 'features/canvas/IAICanvasControls';
import IAICanvasResizer from 'features/canvas/IAICanvasResizer';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import ImageUploadButton from 'common/components/ImageUploaderButton';
import CurrentImageDisplay from 'features/gallery/CurrentImageDisplay';
import { OptionsState } from 'features/options/optionsSlice';
import {
  CanvasState,
  currentCanvasSelector,
  GenericCanvasState,
  OutpaintingCanvasState,
  setDoesCanvasNeedScaling,
} from 'features/canvas/canvasSlice';
import IAICanvas from 'features/canvas/IAICanvas';
import IAICanvasOutpaintingControls from 'features/canvas/IAICanvasOutpaintingControls';

const outpaintingDisplaySelector = createSelector(
  [(state: RootState) => state.canvas, (state: RootState) => state.options],
  (canvas: CanvasState, options: OptionsState) => {
    const {
      doesCanvasNeedScaling,
      outpainting: { objects },
    } = canvas;
    const { showDualDisplay } = options;
    return {
      doesCanvasNeedScaling,
      showDualDisplay,
      doesOutpaintingHaveObjects: objects.length > 0,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const OutpaintingDisplay = () => {
  const dispatch = useAppDispatch();
  const { showDualDisplay, doesCanvasNeedScaling, doesOutpaintingHaveObjects } =
    useAppSelector(outpaintingDisplaySelector);

  useLayoutEffect(() => {
    const resizeCallback = _.debounce(
      () => dispatch(setDoesCanvasNeedScaling(true)),
      250
    );
    window.addEventListener('resize', resizeCallback);
    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  const outpaintingComponent = doesOutpaintingHaveObjects ? (
    <div className="inpainting-main-area">
      <IAICanvasOutpaintingControls />
      <div className="inpainting-canvas-area">
        {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
      </div>
    </div>
  ) : (
    <ImageUploadButton />
  );

  return (
    <div className={'workarea-single-view'}>
      <div className="workarea-split-view-left">{outpaintingComponent}</div>
    </div>
  );
};

export default OutpaintingDisplay;
