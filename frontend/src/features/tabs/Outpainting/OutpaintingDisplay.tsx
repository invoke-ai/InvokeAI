import { createSelector } from '@reduxjs/toolkit';
import IAICanvas from 'features/canvas/IAICanvas';
import IAICanvasControls from 'features/canvas/IAICanvasControls';
import IAICanvasResizer from 'features/canvas/IAICanvasResizer';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import ImageUploadButton from 'common/components/ImageUploaderButton';
import CurrentImageDisplay from 'features/gallery/CurrentImageDisplay';
import { OptionsState } from 'features/options/optionsSlice';
import { OutpaintingState, setDoesCanvasNeedScaling } from './outpaintingSlice';

const inpaintingDisplaySelector = createSelector(
  [(state: RootState) => state.outpainting, (state: RootState) => state.options],
  (outpainting: OutpaintingState, options: OptionsState) => {
    const { doesCanvasNeedScaling, imageToInpaint, boundingBoxDimensions } =
      outpainting;
    const { showDualDisplay } = options;
    return {
      doesCanvasNeedScaling,
      showDualDisplay,
      imageToInpaint,
      boundingBoxDimensions,
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
  const {
    showDualDisplay,
    doesCanvasNeedScaling,
    imageToInpaint,
  } = useAppSelector(inpaintingDisplaySelector);

  useLayoutEffect(() => {
    const resizeCallback = _.debounce(
      () => dispatch(setDoesCanvasNeedScaling(true)),
      250
    );
    window.addEventListener('resize', resizeCallback);
    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  const inpaintingComponent = imageToInpaint ? (
    <div className="inpainting-main-area">
      <IAICanvasControls />
      <div className="inpainting-canvas-area">
        {doesCanvasNeedScaling ? (
          <IAICanvasResizer />
        ) : (
          <IAICanvas />
        )}
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
      <div className="workarea-split-view-left">{inpaintingComponent} </div>
      {showDualDisplay && (
        <div className="workarea-split-view-right">
          <CurrentImageDisplay />
        </div>
      )}
    </div>
  );
};

export default OutpaintingDisplay;
