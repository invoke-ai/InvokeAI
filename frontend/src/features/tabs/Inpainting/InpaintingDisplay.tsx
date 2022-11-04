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
import { InpaintingState, setDoesCanvasNeedScaling } from './inpaintingSlice';

const inpaintingDisplaySelector = createSelector(
  [(state: RootState) => state.inpainting, (state: RootState) => state.options],
  (inpainting: InpaintingState, options: OptionsState) => {
    const { doesCanvasNeedScaling, imageToInpaint } = inpainting;
    const { showDualDisplay } = options;
    return {
      doesCanvasNeedScaling,
      showDualDisplay,
      imageToInpaint,
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
  const { showDualDisplay, doesCanvasNeedScaling, imageToInpaint } =
    useAppSelector(inpaintingDisplaySelector);

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
