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

  let statusStyle;
  if (isConnected && !hasError) {
    statusStyle = 'status-good';
  } else {
    statusStyle = 'status-bad';
  }

  let statusMessage = currentStatus;

  const intermediateStatuses = [
    t('common.statusGenerating'),
    t('common.statusPreparing'),
    t('common.statusSavingImage'),
    t('common.statusRestoringFaces'),
    t('common.statusUpscaling'),
  ];

  if (intermediateStatuses.includes(statusMessage)) {
    statusStyle = 'status-working';
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
        className={`status ${statusStyle}`}
      >
        {t(statusMessage as keyof typeof t)}
      </Text>
    </Tooltip>
  );
};

export default StatusIndicator;
