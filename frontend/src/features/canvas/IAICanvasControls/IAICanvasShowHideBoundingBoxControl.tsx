import { FaVectorSquare } from 'react-icons/fa';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { setShouldShowBoundingBox } from 'features/tabs/Inpainting/inpaintingSlice';

const IAICanvasShowHideBoundingBoxControl = () => {
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

export default IAICanvasShowHideBoundingBoxControl;
