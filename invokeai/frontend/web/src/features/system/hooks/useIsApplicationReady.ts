import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { useMemo } from 'react';

const isApplicationReadySelector = createSelector(
  [(state: RootState) => state.system],
  (system) => {
    const {
      disabledFeatures,
      disabledTabs,
      wereModelsReceived,
      wasSchemaParsed,
    } = system;

    return {
      disabledTabs,
      disabledFeatures,
      wereModelsReceived,
      wasSchemaParsed,
    };
  }
);

export const useIsApplicationReady = () => {
  const {
    disabledTabs,
    disabledFeatures,
    wereModelsReceived,
    wasSchemaParsed,
  } = useAppSelector(isApplicationReadySelector);

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
