import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import ImageUploadButton from '../../../common/components/ImageUploaderButton';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import { OptionsState } from '../../options/optionsSlice';
import InpaintingCanvas from './InpaintingCanvas';
import InpaintingCanvasPlaceholder from './InpaintingCanvasPlaceholder';
import InpaintingControls from './InpaintingControls';
import { InpaintingState, setNeedsCache } from './inpaintingSlice';

const inpaintingDisplaySelector = createSelector(
  [(state: RootState) => state.inpainting, (state: RootState) => state.options],
  (inpainting: InpaintingState, options: OptionsState) => {
    const { needsCache, imageToInpaint } = inpainting;
    const { showDualDisplay } = options;
    return {
      needsCache,
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
  const { showDualDisplay, needsCache, imageToInpaint } = useAppSelector(
    inpaintingDisplaySelector
  );

  useLayoutEffect(() => {
    const resizeCallback = _.debounce(() => dispatch(setNeedsCache(true)), 250);
    window.addEventListener('resize', resizeCallback);
    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  const inpaintingComponent = imageToInpaint ? (
    <div className="inpainting-main-area">
      <InpaintingControls />
      <div className="inpainting-canvas-area">
        {needsCache ? <InpaintingCanvasPlaceholder /> : <InpaintingCanvas />}
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

export default InpaintingDisplay;
