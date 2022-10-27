import { MdCancel } from 'react-icons/md';
import { cancelProcessing } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAIIconButton from '../../../common/components/IAIIconButton';
import { useHotkeys } from 'react-hotkeys-hook';
import { createSelector } from '@reduxjs/toolkit';
import { SystemState } from '../../system/systemSlice';
import _ from 'lodash';

const cancelButtonSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function CancelButton() {
  const dispatch = useAppDispatch();
  const { isProcessing, isConnected } = useAppSelector(cancelButtonSelector);
  const handleClickCancel = () => dispatch(cancelProcessing());

  useHotkeys(
    'shift+x',
    () => {
      if (isConnected || isProcessing) {
        handleClickCancel();
      }
    },
    [isConnected, isProcessing]
  );

  return (
    <IAIIconButton
      icon={<MdCancel />}
      tooltip="Cancel"
      aria-label="Cancel"
      isDisabled={!isConnected || !isProcessing}
      onClick={handleClickCancel}
      styleClass="cancel-btn"
    />
  );
}
