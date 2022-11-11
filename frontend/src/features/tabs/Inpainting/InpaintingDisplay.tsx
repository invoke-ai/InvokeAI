import { createSelector } from '@reduxjs/toolkit';
import IAICanvasControls from 'features/canvas/IAICanvasControls';
import IAICanvasResizer from 'features/canvas/IAICanvasResizer';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import ImageUploadButton from 'common/components/ImageUploaderButton';
import CurrentImageDisplay from 'features/gallery/CurrentImageDisplay';
import { OptionsState } from 'features/options/optionsSlice';
import {
  baseCanvasImageSelector,
  CanvasState,
  setDoesCanvasNeedScaling,
} from 'features/canvas/canvasSlice';
import IAICanvas from 'features/canvas/IAICanvas';

const inpaintingDisplaySelector = createSelector(
  [
    baseCanvasImageSelector,
    (state: RootState) => state.canvas,
    (state: RootState) => state.options,
  ],
  (baseCanvasImage, canvas: CanvasState, options: OptionsState) => {
    const { doesCanvasNeedScaling } = canvas;
    const { showDualDisplay } = options;
    return {
      doesCanvasNeedScaling,
      showDualDisplay,
      baseCanvasImage,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const InpaintingDisplay = () => {
  const dispatch = useAppDispatch();
  const { showDualDisplay, doesCanvasNeedScaling, baseCanvasImage } =
    useAppSelector(inpaintingDisplaySelector);

  useLayoutEffect(() => {
    const resizeCallback = _.debounce(
      () => dispatch(setDoesCanvasNeedScaling(true)),
      250
    );
    window.addEventListener('resize', resizeCallback);
    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  const inpaintingComponent = baseCanvasImage ? (
    <div className="inpainting-main-area">
      <IAICanvasControls />
      <div className="inpainting-canvas-area">
        {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
      </div>
    </div>
  ) : (
    <ImageUploadButton />
  );

  return (
    <div
      className={
        showDualDisplay ? 'workarea-split-view' : 'workarea-single-view'
      }
    >
      <div className="workarea-split-view-left">{inpaintingComponent}</div>
      {showDualDisplay && (
        <div className="workarea-split-view-right">
          <CurrentImageDisplay />
        </div>
      )}
    </div>
  );
};

export default InpaintingDisplay;
