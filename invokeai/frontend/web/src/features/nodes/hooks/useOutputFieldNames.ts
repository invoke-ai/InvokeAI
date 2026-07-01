import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import { getSortedFilteredFieldNames } from 'features/nodes/util/node/getSortedFilteredFieldNames';
import { useMemo } from 'react';

export const useOutputFieldNames = (): string[] => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector([ctx.selectNodeTemplateOrThrow], (template) =>
        getSortedFilteredFieldNames(Object.values(template.outputs))
      ),
    [ctx]
  );
  return useAppSelector(selector);
};
