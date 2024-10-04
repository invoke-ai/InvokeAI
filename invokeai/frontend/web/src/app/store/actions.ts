import { createAction } from '@reduxjs/toolkit';
import type { TabName } from 'features/ui/store/uiTypes';

export const enqueueRequested = createAction<{
  tabName: TabName;
  prepend: boolean;
}>('app/enqueueRequested');
