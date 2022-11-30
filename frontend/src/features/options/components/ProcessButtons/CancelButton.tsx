import { MdCancel } from 'react-icons/md';
import { cancelProcessing } from 'app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { useHotkeys } from 'react-hotkeys-hook';
import { createSelector } from '@reduxjs/toolkit';
import { SystemState } from 'features/system/store/systemSlice';
import _ from 'lodash';

const cancelButtonSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
      isCancelable: system.isCancelable,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function CancelButton(
  props: Omit<IAIIconButtonProps, 'aria-label'>
) {
  const { ...rest } = props;
  const dispatch = useAppDispatch();
  const { isProcessing, isConnected, isCancelable } =
    useAppSelector(cancelButtonSelector);
  const handleClickCancel = () => dispatch(cancelProcessing());

  useHotkeys(
    'shift+x',
    () => {
      if ((isConnected || isProcessing) && isCancelable) {
        handleClickCancel();
      }
    },
    [isConnected, isProcessing, isCancelable]
  );

  return (
    <IAIIconButton
      icon={<MdCancel />}
      tooltip="Cancel"
      aria-label="Cancel"
      isDisabled={!isConnected || !isProcessing || !isCancelable}
      onClick={handleClickCancel}
      styleClass="cancel-btn"
      {...rest}
    />
  );
}
