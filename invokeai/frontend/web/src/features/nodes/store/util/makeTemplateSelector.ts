import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { InvocationTemplate } from 'features/nodes/types/types';
import { AnyInvocationType } from 'services/events/types';

export const makeTemplateSelector = (type: AnyInvocationType) =>
  createSelector(
    [(state: RootState) => state.nodes],
    (nodes) => {
      const template = nodes.invocationTemplates[type];
      if (!template) {
        return;
      }
      return template;
    },
    {
      memoizeOptions: {
        resultEqualityCheck: (
          a: InvocationTemplate | undefined,
          b: InvocationTemplate | undefined
        ) => a !== undefined && b !== undefined && a.type === b.type,
      },
    }
  );
