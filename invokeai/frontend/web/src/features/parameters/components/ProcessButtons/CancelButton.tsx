import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  SystemState,
  cancelScheduled,
  cancelTypeChanged,
  CancelStrategy,
} from 'features/system/store/systemSlice';
import { isEqual } from 'lodash-es';
import { useCallback, memo } from 'react';
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
import { MdCancel, MdCancelScheduleSend } from 'react-icons/md';

import { sessionCanceled } from 'services/thunks/session';
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
    cancelType,
    isCancelScheduled,
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
      dispatch(cancelTypeChanged(newCancelType as CancelStrategy));
    },
    [dispatch]
  );

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
          as={IAIIconButton}
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
