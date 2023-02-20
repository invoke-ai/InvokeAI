import { createSelector } from '@reduxjs/toolkit';
import { cancelProcessing } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  SystemState,
  setCancelAfter,
  setCancelType,
} from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { useEffect, useCallback } from 'react';
import { ButtonSpinner, ButtonGroup } from '@chakra-ui/react';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { MdCancel, MdCancelScheduleSend } from 'react-icons/md';

import IAISimpleMenu from 'common/components/IAISimpleMenu';

const cancelButtonSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
      isCancelable: system.isCancelable,
      currentIteration: system.currentIteration,
      totalIterations: system.totalIterations,
      cancelType: system.cancelOptions.cancelType,
      cancelAfter: system.cancelOptions.cancelAfter,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

interface CancelButtonProps {
  btnGroupWidth?: string | number;
}

export default function CancelButton(
  props: CancelButtonProps & Omit<IAIIconButtonProps, 'aria-label'>
) {
  const dispatch = useAppDispatch();
  const { btnGroupWidth = 'auto', ...rest } = props;
  const {
    isProcessing,
    isConnected,
    isCancelable,
    currentIteration,
    totalIterations,
    cancelType,
    cancelAfter,
  } = useAppSelector(cancelButtonSelector);
  const handleClickCancel = useCallback(() => {
    dispatch(cancelProcessing());
    dispatch(setCancelAfter(null));
  }, [dispatch]);

  const { t } = useTranslation();

  const isCancelScheduled = cancelAfter === null ? false : true;

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
    if (cancelAfter !== null && cancelAfter < currentIteration) {
      handleClickCancel();
    }
  }, [cancelAfter, currentIteration, handleClickCancel]);

  const cancelMenuItems = [
    {
      item: t('parameters.cancel.immediate'),
      onClick: () => dispatch(setCancelType('immediate')),
    },
    {
      item: t('parameters.cancel.schedule'),
      onClick: () => dispatch(setCancelType('scheduled')),
    },
  ];

  return (
    <ButtonGroup
      isAttached
      variant="link"
      minHeight="2.5rem"
      width={btnGroupWidth}
    >
      {cancelType === 'immediate' ? (
        <IAIIconButton
          icon={<MdCancel />}
          tooltip={t('parameters.cancel.immediate')}
          aria-label={t('parameters.cancel.immediate')}
          isDisabled={!isConnected || !isProcessing || !isCancelable}
          onClick={handleClickCancel}
          className="cancel-btn"
          {...rest}
        />
      ) : (
        <IAIIconButton
          icon={
            isCancelScheduled ? (
              <ButtonSpinner color="var(--text-color)" />
            ) : (
              <MdCancelScheduleSend />
            )
          }
          tooltip={
            isCancelScheduled
              ? t('parameters.cancel.isScheduled')
              : t('parameters.cancel.schedule')
          }
          aria-label={
            isCancelScheduled
              ? t('parameters.cancel.isScheduled')
              : t('parameters.cancel.schedule')
          }
          isDisabled={
            !isConnected ||
            !isProcessing ||
            !isCancelable ||
            currentIteration === totalIterations
          }
          onClick={() => {
            // If a cancel request has already been made, and the user clicks again before the next iteration has been processed, stop the request.
            if (isCancelScheduled) dispatch(setCancelAfter(null));
            else dispatch(setCancelAfter(currentIteration));
          }}
          className="cancel-btn"
          {...rest}
        />
      )}
      <IAISimpleMenu
        menuItems={cancelMenuItems}
        iconTooltip={t('parameters.cancel.setType')}
        menuButtonProps={{
          backgroundColor: 'var(--destructive-color)',
          color: 'var(--text-color)',
          minWidth: '1.5rem',
          minHeight: '1.5rem',
          _hover: {
            backgroundColor: 'var(--destructive-color-hover)',
          },
        }}
      />
    </ButtonGroup>
  );
}
