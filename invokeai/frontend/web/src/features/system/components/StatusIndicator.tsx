import { Text, Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { errorSeen, SystemState } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { useTranslation } from 'react-i18next';
import { systemSelector } from '../store/systemSelectors';

const statusIndicatorSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      isConnected: system.isConnected,
      isProcessing: system.isProcessing,
      currentIteration: system.currentIteration,
      totalIterations: system.totalIterations,
      currentStatus: system.currentStatus,
      hasError: system.hasError,
      wasErrorSeen: system.wasErrorSeen,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const StatusIndicator = () => {
  const {
    isConnected,
    isProcessing,
    currentIteration,
    totalIterations,
    currentStatus,
    hasError,
    wasErrorSeen,
  } = useAppSelector(statusIndicatorSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  let statusIdentifier;

  if (isConnected && !hasError) {
    statusIdentifier = 'ok';
  } else {
    statusIdentifier = 'error';
  }

  let statusMessage = currentStatus;

  if (isProcessing) {
    statusIdentifier = 'working';
  }

  if (statusMessage)
    if (isProcessing) {
      if (totalIterations > 1) {
        statusMessage = `${t(
          statusMessage as keyof typeof t
        )} (${currentIteration}/${totalIterations})`;
      }
    }

  const tooltipLabel =
    hasError && !wasErrorSeen
      ? 'Click to clear, check logs for details'
      : undefined;

  const statusIndicatorCursor =
    hasError && !wasErrorSeen ? 'pointer' : 'initial';

  const handleClickStatusIndicator = () => {
    if (hasError || !wasErrorSeen) {
      dispatch(errorSeen());
    }
  };

  return (
    <Tooltip label={tooltipLabel}>
      <Text
        cursor={statusIndicatorCursor}
        onClick={handleClickStatusIndicator}
        sx={{
          fontSize: 'xs',
          fontWeight: '600',
          color: `${statusIdentifier}.400`,
        }}
      >
        {t(statusMessage as keyof typeof t)}
      </Text>
    </Tooltip>
  );
};

export default StatusIndicator;
