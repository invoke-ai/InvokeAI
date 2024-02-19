import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { InvocationTemplate } from 'features/nodes/types/invocation';

import type { NodeTemplatesState } from './types';

export const initialNodeTemplatesState: NodeTemplatesState = {
  templates: {},
};

export const nodesTemplatesSlice = createSlice({
  name: 'nodeTemplates',
  initialState: initialNodeTemplatesState,
  reducers: {
    nodeTemplatesBuilt: (state, action: PayloadAction<Record<string, InvocationTemplate>>) => {
      state.templates = action.payload;
    },
  },
});

export const { nodeTemplatesBuilt } = nodesTemplatesSlice.actions;

export const selectNodeTemplatesSlice = (state: RootState) => state.nodeTemplates;
