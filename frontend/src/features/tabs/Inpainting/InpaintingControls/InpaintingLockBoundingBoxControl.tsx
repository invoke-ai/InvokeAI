import { FaLock, FaUnlock } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAIIconButton from '../../../../common/components/IAIIconButton';
import { setShouldLockBoundingBox } from '../inpaintingSlice';

const InpaintingLockBoundingBoxControl = () => {
  const dispatch = useAppDispatch();
  const shouldLockBoundingBox = useAppSelector(
    (state: RootState) => state.inpainting.shouldLockBoundingBox
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

export default InpaintingLockBoundingBoxControl;
