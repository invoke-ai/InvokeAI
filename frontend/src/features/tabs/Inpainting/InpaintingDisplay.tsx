import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import { OptionsState } from '../../options/optionsSlice';
import InpaintingCanvas from './InpaintingCanvas';
import InpaintingCanvasPlaceholder from './InpaintingCanvasPlaceholder';
import InpaintingControls from './InpaintingControls';
import { InpaintingState, setNeedsRepaint } from './inpaintingSlice';

const inpaintingDisplaySelector = createSelector(
  [(state: RootState) => state.inpainting, (state: RootState) => state.options],
  (inpainting: InpaintingState, options: OptionsState) => {
    const { needsRepaint } = inpainting;
    const { showDualDisplay } = options;
    return {
      needsRepaint,
      showDualDisplay,
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
  const { showDualDisplay, needsRepaint } = useAppSelector(
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

  return (
    <div
      className="inpainting-display"
      style={
        showDualDisplay
          ? { gridTemplateColumns: '1fr 1fr' }
          : { gridTemplateColumns: 'auto' }
      }
    >
      <div className="inpainting-toolkit">
        <InpaintingControls />

        <div className="inpainting-canvas-container">
          {needsRepaint ? (
            <InpaintingCanvasPlaceholder />
          ) : (
            <InpaintingCanvas />
          )}
        </div>
      </div>
      {showDualDisplay && <CurrentImageDisplay />}
    </div>
  );
};

export default InpaintingDisplay;
