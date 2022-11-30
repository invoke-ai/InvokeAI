import { emptyTempFolder } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import {
  clearCanvasHistory,
  resetCanvas,
} from 'features/canvas/store/canvasSlice';
import { FaTrash } from 'react-icons/fa';

const EmptyTempFolderButtonModal = () => {
  const isStaging = useAppSelector(isStagingSelector);
  const dispatch = useAppDispatch();

  const acceptCallback = () => {
    dispatch(emptyTempFolder());
    dispatch(resetCanvas());
    dispatch(clearCanvasHistory());
  };

  return (
    <IAIAlertDialog
      title={'Empty Temp Image Folder'}
      acceptCallback={acceptCallback}
      acceptButtonText={'Empty Folder'}
      triggerComponent={
        <IAIButton leftIcon={<FaTrash />} size={'sm'} isDisabled={isStaging}>
          Empty Temp Image Folder
        </IAIButton>
      }
    >
      <p>
        Emptying the temp image folder also fully resets the Unified Canvas.
        This includes all undo/redo history, images in the staging area, and the
        canvas base layer.
      </p>
      <br />
      <p>Are you sure you want to empty the temp folder?</p>
    </IAIAlertDialog>
  );
};
export default EmptyTempFolderButtonModal;
