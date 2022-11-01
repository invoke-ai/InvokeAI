import React from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../../app/store';
import IAICheckbox from '../../../../../common/components/IAICheckbox';
import { setShouldShowBoundingBoxFill } from '../../../../tabs/Inpainting/inpaintingSlice';

export default function BoundingBoxDarkenOutside() {
  const dispatch = useAppDispatch();
  const shouldShowBoundingBoxFill = useAppSelector(
    (state: RootState) => state.inpainting.shouldShowBoundingBoxFill
  );

  const handleChangeShouldShowBoundingBoxFill = () => {
    dispatch(setShouldShowBoundingBoxFill(!shouldShowBoundingBoxFill));
  };

  return (
    <IAICheckbox
      label="Darken Outside Box"
      isChecked={shouldShowBoundingBoxFill}
      onChange={handleChangeShouldShowBoundingBoxFill}
      styleClass="inpainting-bounding-box-darken"
    />
  );
}
