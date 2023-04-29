import { Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';
import { systemSelector } from '../store/systemSelectors';

const statusIndicatorSelector = createSelector(
  systemSelector,
  (system) => {
    return {
      isConnected: system.isConnected,
      isProcessing: system.isProcessing,
      currentIteration: system.currentIteration,
      totalIterations: system.totalIterations,
      currentStatus: system.currentStatus,
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
  } = useAppSelector(statusIndicatorSelector);
  const { t } = useTranslation();

  let statusIdentifier;

  if (isConnected) {
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

  return (
    <Text
      sx={{
        fontSize: 'sm',
        fontWeight: '600',
        color: `${statusIdentifier}.400`,
      }}
    >
      {t(statusMessage as keyof typeof t)}
    </Text>
  );
};

export default StatusIndicator;
