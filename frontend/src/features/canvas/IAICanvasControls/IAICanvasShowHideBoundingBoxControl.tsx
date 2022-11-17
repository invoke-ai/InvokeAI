import { FaVectorSquare } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  canvasSelector,
  setShouldShowBoundingBox,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';

const canvasShowHideBoundingBoxControlSelector = createSelector(
  canvasSelector,
  (canvas) => {
    const { shouldShowBoundingBox } = canvas;

    return {
      shouldShowBoundingBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);
const IAICanvasShowHideBoundingBoxControl = () => {
  const dispatch = useAppDispatch();
  const { shouldShowBoundingBox } = useAppSelector(
    canvasShowHideBoundingBoxControlSelector
  );

  return (
    <IAIIconButton
      aria-label="Hide Inpainting Box (Shift+H)"
      tooltip="Hide Inpainting Box (Shift+H)"
      icon={<FaVectorSquare />}
      data-alert={!shouldShowBoundingBox}
      onClick={() => {
        dispatch(setShouldShowBoundingBox(!shouldShowBoundingBox));
      }}
    />
  );
};

export default IAICanvasShowHideBoundingBoxControl;
