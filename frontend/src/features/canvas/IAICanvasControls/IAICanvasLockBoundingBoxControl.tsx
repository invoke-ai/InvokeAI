import { FaLock, FaUnlock } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setShouldLockBoundingBox,
} from 'features/canvas/canvasSlice';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';

const canvasLockBoundingBoxSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas: GenericCanvasState) => {
    const { shouldLockBoundingBox } = currentCanvas;

    return {
      shouldLockBoundingBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const IAICanvasLockBoundingBoxControl = () => {
  const dispatch = useAppDispatch();
  const { shouldLockBoundingBox } = useAppSelector(
    canvasLockBoundingBoxSelector
  );

  return (
    <IAIIconButton
      aria-label="Lock Inpainting Box"
      tooltip="Lock Inpainting Box"
      icon={shouldLockBoundingBox ? <FaLock /> : <FaUnlock />}
      data-selected={shouldLockBoundingBox}
      onClick={() => {
        dispatch(setShouldLockBoundingBox(!shouldLockBoundingBox));
      }}
    />
  );
};

export default IAICanvasLockBoundingBoxControl;
