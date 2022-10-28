import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { uploadImage } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import InvokeImageUploader from '../../../common/components/InvokeImageUploader';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import { OptionsState } from '../../options/optionsSlice';
import InpaintingCanvas from './InpaintingCanvas';
import InpaintingCanvasPlaceholder from './InpaintingCanvasPlaceholder';
import InpaintingControls from './InpaintingControls';
import { InpaintingState, setNeedsRepaint } from './inpaintingSlice';

const inpaintingDisplaySelector = createSelector(
  [(state: RootState) => state.inpainting, (state: RootState) => state.options],
  (inpainting: InpaintingState, options: OptionsState) => {
    const { needsRepaint, imageToInpaint } = inpainting;
    const { showDualDisplay } = options;
    return {
      needsRepaint,
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
  const { showDualDisplay, needsRepaint, imageToInpaint } = useAppSelector(
    inpaintingDisplaySelector
  );

  useLayoutEffect(() => {
    const resizeCallback = _.debounce(
      () => dispatch(setNeedsRepaint(true)),
      250
    );
    window.addEventListener('resize', resizeCallback);
    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  const inpaintingComponent = imageToInpaint ? (
    <div className="inpainting-main-area">
      <InpaintingControls />
      <div className="inpainting-canvas-area">
        {needsRepaint ? <InpaintingCanvasPlaceholder /> : <InpaintingCanvas />}
      </div>
    </div>
  ) : (
    <InvokeImageUploader
      handleFile={(file: File) =>
        dispatch(uploadImage({ file, destination: 'inpainting' }))
      }
    />
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
