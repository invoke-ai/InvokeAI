import { createAction } from '@reduxjs/toolkit';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import type { BatchConfig } from 'services/api/types';

export const enqueueRequested = createAction<{
  tabName: InvokeTabName;
  prepend: boolean;
}>('app/enqueueRequested');

export const batchEnqueued = createAction<BatchConfig>('app/batchEnqueued');
