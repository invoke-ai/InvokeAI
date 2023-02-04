import { createSelector } from '@reduxjs/toolkit';
import { cancelProcessing } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { systemSelector } from 'features/system/store/systemSelectors';
import { SystemState } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { MdCancel } from 'react-icons/md';

const cancelButtonSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
      isCancelable: system.isCancelable,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
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
      tooltip={t('parameters:cancel')}
      aria-label={t('parameters:cancel')}
      isDisabled={!isConnected || !isProcessing || !isCancelable}
      onClick={handleClickCancel}
      styleClass="cancel-btn"
      {...rest}
    />
  );
}
