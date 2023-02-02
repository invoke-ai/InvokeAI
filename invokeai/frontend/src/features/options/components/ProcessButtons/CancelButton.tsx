import { MdCancel } from 'react-icons/md';
import { cancelProcessing } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { useHotkeys } from 'react-hotkeys-hook';
import { createSelector } from '@reduxjs/toolkit';
import { SystemState } from 'features/system/store/systemSlice';
import _ from 'lodash';
import { useTranslation } from 'react-i18next';

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

  const { t } = useTranslation();

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
      tooltip={t('options:cancel')}
      aria-label={t('options:cancel')}
      isDisabled={!isConnected || !isProcessing || !isCancelable}
      onClick={handleClickCancel}
      styleClass="cancel-btn"
      {...rest}
    />
  );
}
