import { FaVectorSquare } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIIconButton from '../../../../common/components/IAIIconButton';
import { setShouldShowBoundingBox } from '../inpaintingSlice';

const InpaintingShowHideBoundingBoxControl = () => {
  const dispatch = useAppDispatch();
  const shouldShowBoundingBox = useAppSelector(
    (state: RootState) => state.inpainting.shouldShowBoundingBox
  );

  return (
    <IAIIconButton
      aria-label="Hide Inpainting Box"
      tooltip="Hide Inpainting Box"
      icon={<FaVectorSquare />}
      data-alert={!shouldShowBoundingBox}
      onClick={() => {
        dispatch(setShouldShowBoundingBox(!shouldShowBoundingBox));
      }}
    />
  );
};

export default InpaintingShowHideBoundingBoxControl;
