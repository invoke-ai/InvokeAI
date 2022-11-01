import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../../app/store';
import IAICheckbox from '../../../../../common/components/IAICheckbox';
import { setShouldLockBoundingBox } from '../../../../tabs/Inpainting/inpaintingSlice';

export default function BoundingBoxLock() {
  const shouldLockBoundingBox = useAppSelector(
    (state: RootState) => state.inpainting.shouldLockBoundingBox
  );
  const dispatch = useAppDispatch();

  const handleChangeShouldLockBoundingBox = () => {
    dispatch(setShouldLockBoundingBox(!shouldLockBoundingBox));
  };
  return (
    <IAICheckbox
      label="Lock Bounding Box"
      isChecked={shouldLockBoundingBox}
      onChange={handleChangeShouldLockBoundingBox}
      styleClass="inpainting-bounding-box-darken"
    />
  );
}
