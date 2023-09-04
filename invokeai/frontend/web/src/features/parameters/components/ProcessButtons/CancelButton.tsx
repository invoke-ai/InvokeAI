import {
  ButtonGroup,
  ButtonProps,
  ButtonSpinner,
  Menu,
  MenuButton,
  MenuItemOption,
  MenuList,
  MenuOptionGroup,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { systemSelector } from 'features/system/store/systemSelectors';
import {
  CancelStrategy,
  SystemState,
  cancelScheduled,
  cancelTypeChanged,
} from 'features/system/store/systemSlice';
import { isEqual } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';

import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { MdCancel, MdCancelScheduleSend } from 'react-icons/md';

import { ChevronDownIcon } from '@chakra-ui/icons';
import { sessionCanceled } from 'services/api/thunks/session';
import IAIButton from 'common/components/IAIButton';

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

type Props = Omit<ButtonProps, 'aria-label'> & {
  btnGroupWidth?: string | number;
  asIconButton?: boolean;
};

const CancelButton = (props: Props) => {
  const dispatch = useAppDispatch();
  const { btnGroupWidth = 'auto', asIconButton = false, ...rest } = props;
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

    dispatch(sessionCanceled({ session_id: sessionId }));
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

  const cancelLabel = useMemo(() => {
    if (isCancelScheduled) {
      return t('parameters.cancel.isScheduled');
    }
    if (cancelType === 'immediate') {
      return t('parameters.cancel.immediate');
    }

    return t('parameters.cancel.schedule');
  }, [t, cancelType, isCancelScheduled]);

  const cancelIcon = useMemo(() => {
    if (isCancelScheduled) {
      return <ButtonSpinner />;
    }
    if (cancelType === 'immediate') {
      return <MdCancel />;
    }

    return <MdCancelScheduleSend />;
  }, [cancelType, isCancelScheduled]);

  return (
    <ButtonGroup isAttached width={btnGroupWidth}>
      {asIconButton ? (
        <IAIIconButton
          icon={cancelIcon}
          tooltip={cancelLabel}
          aria-label={cancelLabel}
          isDisabled={!isConnected || !isProcessing || !isCancelable}
          onClick={handleClickCancel}
          colorScheme="error"
          id="cancel-button"
          {...rest}
        />
      ) : (
        <IAIButton
          leftIcon={cancelIcon}
          tooltip={cancelLabel}
          aria-label={cancelLabel}
          isDisabled={!isConnected || !isProcessing || !isCancelable}
          onClick={handleClickCancel}
          colorScheme="error"
          id="cancel-button"
          {...rest}
        >
          Cancel
        </IAIButton>
      )}
      <Menu closeOnSelect={false}>
        <MenuButton
          as={IAIIconButton}
          tooltip={t('parameters.cancel.setType')}
          aria-label={t('parameters.cancel.setType')}
          icon={<ChevronDownIcon w="1em" h="1em" />}
          paddingX={0}
          paddingY={0}
          colorScheme="error"
          minWidth={5}
          {...rest}
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
