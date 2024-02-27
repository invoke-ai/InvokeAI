import { createAction } from '@reduxjs/toolkit';
import type { InvokeTabName } from 'features/ui/store/tabMap';

export const enqueueRequested = createAction<{
  tabName: InvokeTabName;
  prepend: boolean;
}>('app/enqueueRequested');
