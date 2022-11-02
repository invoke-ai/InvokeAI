import React from 'react';
import { BiHide, BiShow } from 'react-icons/bi';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../../app/store';
import IAIIconButton from '../../../../../common/components/IAIIconButton';
import { setShouldShowBoundingBox } from '../../../../tabs/Inpainting/inpaintingSlice';

export default function BoundingBoxVisibility() {
  const shouldShowBoundingBox = useAppSelector(
    (state: RootState) => state.inpainting.shouldShowBoundingBox
  );
  const dispatch = useAppDispatch();

  const handleShowBoundingBox = () =>
    dispatch(setShouldShowBoundingBox(!shouldShowBoundingBox));
  return (
    <IAIIconButton
      aria-label="Toggle Bounding Box Visibility"
      icon={shouldShowBoundingBox ? <BiShow size={22} /> : <BiHide size={22} />}
      onClick={handleShowBoundingBox}
      background={'none'}
      padding={0}
    />
  );
}
