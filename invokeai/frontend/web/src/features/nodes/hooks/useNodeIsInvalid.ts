import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { getInvocationNodeErrors } from 'features/nodes/store/util/fieldValidators';
import { useMemo } from 'react';

export const useNodeIsInvalid = (nodeId: string) => {
  const templates = useStore($templates);
  const selectHasErrors = useMemo(
    () =>
      createSelector(selectNodesSlice, (nodes) => {
        const errors = getInvocationNodeErrors(nodeId, templates, nodes);
        return errors.length > 0;
      }),
    [nodeId, templates]
  );
  const hasErrors = useAppSelector(selectHasErrors);
  return hasErrors;
};
