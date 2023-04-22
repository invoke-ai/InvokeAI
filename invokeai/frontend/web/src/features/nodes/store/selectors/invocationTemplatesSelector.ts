import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';

export const invocationTemplatesSelector = createSelector(
  (state: RootState) => state.nodes,
  (nodes) => nodes.invocationTemplates
);
