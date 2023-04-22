import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  SystemState,
  setCancelAfter,
  setCancelType,
  cancelScheduled,
  cancelTypeChanged,
  CancelType,
} from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { useEffect, useCallback, memo } from 'react';
import {
  ButtonSpinner,
  ButtonGroup,
  Menu,
  MenuButton,
  MenuList,
  MenuOptionGroup,
  MenuItemOption,
  IconButton,
} from '@chakra-ui/react';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import {
  MdArrowDropDown,
  MdArrowDropUp,
  MdCancel,
  MdCancelScheduleSend,
} from 'react-icons/md';

import IAISimpleMenu from 'common/components/IAISimpleMenu';
import { sessionCanceled } from 'services/thunks/session';
import { FaChevronDown } from 'react-icons/fa';
import { BiChevronDown } from 'react-icons/bi';

const cancelButtonSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
      isCancelable: system.isCancelable,
      currentIteration: system.currentIteration,
      totalIterations: system.totalIterations,
      // cancelType: system.cancelOptions.cancelType,
      // cancelAfter: system.cancelOptions.cancelAfter,
      sessionId: system.sessionId,
      cancelType: system.cancelType,
      isCancelScheduled: system.isCancelScheduled,
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

const CancelButton = (
  props: CancelButtonProps & Omit<IAIIconButtonProps, 'aria-label'>
) => {
  const dispatch = useAppDispatch();
  const { btnGroupWidth = 'auto', ...rest } = props;
  const {
    isProcessing,
    isConnected,
    isCancelable,
    currentIteration,
    totalIterations,
    cancelType,
    isCancelScheduled,
    // cancelAfter,
    sessionId,
  } = useAppSelector(cancelButtonSelector);

  const handleClickCancel = useCallback(() => {
    if (!sessionId) {
      return;
    }

    if (cancelType === 'scheduled') {
      dispatch(cancelScheduled());
      return;
    }

    dispatch(sessionCanceled({ sessionId }));
  }, [dispatch, sessionId, cancelType]);

  const { t } = useTranslation();

  const handleCancelTypeChanged = useCallback(
    (value: string | string[]) => {
      const newCancelType = Array.isArray(value) ? value[0] : value;
      dispatch(cancelTypeChanged(newCancelType as CancelType));
    },
    [dispatch]
  );
  // const isCancelScheduled = cancelAfter === null ? false : true;

  useHotkeys(
    'shift+x',
    () => {
      if ((isConnected || isProcessing) && isCancelable) {
        handleClickCancel();
      }
    },
    [isConnected, isProcessing, isCancelable]
  );

  // useEffect(() => {
  //   if (cancelAfter !== null && cancelAfter < currentIteration) {
  //     handleClickCancel();
  //   }
  // }, [cancelAfter, currentIteration, handleClickCancel]);

  // const cancelMenuItems = [
  //   {
  //     item: t('parameters.cancel.immediate'),
  //     onClick: () => dispatch(cancelTypeChanged('immediate')),
  //   },
  //   {
  //     item: t('parameters.cancel.schedule'),
  //     onClick: () => dispatch(cancelTypeChanged('scheduled')),
  //   },
  // ];

  return (
    <ButtonGroup isAttached width={btnGroupWidth}>
      {cancelType === 'immediate' ? (
        <IAIIconButton
          icon={<MdCancel />}
          tooltip={t('parameters.cancel.immediate')}
          aria-label={t('parameters.cancel.immediate')}
          isDisabled={!isConnected || !isProcessing || !isCancelable}
          onClick={handleClickCancel}
          colorScheme="error"
          {...rest}
        />
      ) : (
        <IAIIconButton
          icon={
            isCancelScheduled ? <ButtonSpinner /> : <MdCancelScheduleSend />
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
          isDisabled={!isConnected || !isProcessing || !isCancelable}
          onClick={handleClickCancel}
          colorScheme="error"
          {...rest}
        />
      )}

      <Menu closeOnSelect={false}>
        <MenuButton
          as={IconButton}
          tooltip={t('parameters.cancel.setType')}
          aria-label={t('parameters.cancel.setType')}
          icon={<BiChevronDown />}
          paddingX={0}
          paddingY={0}
          colorScheme="error"
          minWidth={5}
        />
        <MenuList minWidth="240px">
          <MenuOptionGroup
            value={cancelType}
            title="Cancel Type"
            type="radio"
            onChange={handleCancelTypeChanged}
          >
            <MenuItemOption value="immediate">
              {t('parameters.cancel.immediate')}
            </MenuItemOption>
            <MenuItemOption value="scheduled">
              {t('parameters.cancel.schedule')}
            </MenuItemOption>
          </MenuOptionGroup>
        </MenuList>
      </Menu>
    </ButtonGroup>
  );
};

export default memo(CancelButton);
