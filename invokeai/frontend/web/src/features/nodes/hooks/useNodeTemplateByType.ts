import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useMemo } from 'react';
import { AnyInvocationType } from 'services/events/types';

export const useNodeTemplateByType = (
  type: AnyInvocationType | 'current_image' | 'notes'
) => {
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => {
          const nodeTemplate = nodes.nodeTemplates[type];
          return nodeTemplate;
        },
        defaultSelectorOptions
      ),
    [type]
  );

  const nodeTemplate = useAppSelector(selector);

  return nodeTemplate;
};
