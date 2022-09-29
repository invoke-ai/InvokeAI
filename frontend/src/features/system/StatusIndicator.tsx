import { Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { RootState, useAppSelector } from '../../app/store';
import { SystemState } from './systemSlice';

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
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
  } = useAppSelector(systemSelector);
  const statusMessageTextColor = isConnected ? 'green.500' : 'red.500';

  let statusMessage = currentStatus;

  if (isProcessing) {
    if (totalIterations > 1) {
      statusMessage += ` (${currentIteration}/${totalIterations})`;
    }
  }
  return <Text textColor={statusMessageTextColor}>{statusMessage}</Text>;
};

export default StatusIndicator;
