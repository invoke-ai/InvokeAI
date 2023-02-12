import { createSelector } from '@reduxjs/toolkit';
import { cancelProcessing } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { systemSelector } from 'features/system/store/systemSelectors';
import { SystemState } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { useState, useEffect, useCallback } from 'react';
import { Spinner } from '@chakra-ui/react';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { MdCancel, MdCancelScheduleSend } from 'react-icons/md';

const cancelButtonSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
      isCancelable: system.isCancelable,
      currentIteration: system.currentIteration,
      totalIterations: system.totalIterations,
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
  const dispatch = useAppDispatch();
  const { ...rest } = props;
  const {
    isProcessing,
    isConnected,
    isCancelable,
    currentIteration,
    totalIterations,
  } = useAppSelector(cancelButtonSelector);
  const [cancelAfterIteration, setCancelAfterIteration] = useState<number>(
    Number.MAX_SAFE_INTEGER
  );
  const handleClickCancel = useCallback(() => {
    dispatch(cancelProcessing());
    setCancelAfterIteration(Number.MAX_SAFE_INTEGER);
  }, [dispatch]);

  const { t } = useTranslation();

  const isCancelScheduled = cancelAfterIteration !== Number.MAX_SAFE_INTEGER;

  useHotkeys(
    'shift+x',
    () => {
      if ((isConnected || isProcessing) && isCancelable) {
        handleClickCancel();
      }
    },
    [isConnected, isProcessing, isCancelable]
  );

  useEffect(() => {
    if (cancelAfterIteration < currentIteration) {
      handleClickCancel();
    }
  }, [cancelAfterIteration, currentIteration, handleClickCancel]);

  return (
    <>
      <IAIIconButton
        icon={<MdCancel />}
        tooltip={t('parameters:cancel')}
        aria-label={t('parameters:cancel')}
        isDisabled={!isConnected || !isProcessing || !isCancelable}
        onClick={handleClickCancel}
        styleClass="cancel-btn"
        {...rest}
      />
      <IAIIconButton
        icon={isCancelScheduled ? <Spinner /> : <MdCancelScheduleSend />}
        tooltip={
          isCancelScheduled
            ? t('parameters:cancelScheduled')
            : t('parameters:scheduleCancel')
        }
        aria-label={
          isCancelScheduled
            ? t('parameters:cancelScheduled')
            : t('parameters:scheduleCancel')
        }
        isDisabled={
          !isConnected ||
          !isProcessing ||
          !isCancelable ||
          currentIteration === totalIterations
        }
        onClick={() => {
          // If a cancel request has already been made, and the user clicks again before the next iteration has been processed, stop the request.
          if (isCancelScheduled)
            setCancelAfterIteration(Number.MAX_SAFE_INTEGER);
          else setCancelAfterIteration(currentIteration);
        }}
        styleClass="cancel-btn"
        {...rest}
      />
    </>
  );
}
