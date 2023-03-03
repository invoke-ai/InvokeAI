import { Progress } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { SystemState } from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { systemSelector } from '../store/systemSelectors';

const progressBarSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      currentStep: system.currentStep,
      totalSteps: system.totalSteps,
      currentStatusHasSteps: system.currentStatusHasSteps,
    };
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const ProgressBar = () => {
  const { isProcessing, currentStep, totalSteps, currentStatusHasSteps } =
    useAppSelector(progressBarSelector);

  const value = currentStep ? Math.round((currentStep * 100) / totalSteps) : 0;

  return (
    <Progress
      value={value}
      isIndeterminate={isProcessing && !currentStatusHasSteps}
      className="progress-bar"
    />
  );
};

export default ProgressBar;
