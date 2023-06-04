import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useMemo } from 'react';
import { configSelector } from '../store/configSelectors';
import { systemSelector } from '../store/systemSelectors';

const isApplicationReadySelector = createSelector(
  [systemSelector, configSelector],
  (system, config) => {
    const { wereModelsReceived, wasSchemaParsed } = system;

    const { disabledTabs } = config;

    return {
      disabledTabs,
      wereModelsReceived,
      wasSchemaParsed,
    };
  }
);

/**
 * Checks if the application is ready to be used, i.e. if the initial startup process is finished.
 */
export const useIsApplicationReady = () => {
  const { disabledTabs, wereModelsReceived, wasSchemaParsed } = useAppSelector(
    isApplicationReadySelector
  );

  const isApplicationReady = useMemo(() => {
    if (!wereModelsReceived) {
      return false;
    }

    if (!disabledTabs.includes('nodes') && !wasSchemaParsed) {
      return false;
    }

    return true;
  }, [disabledTabs, wereModelsReceived, wasSchemaParsed]);

  return isApplicationReady;
};
