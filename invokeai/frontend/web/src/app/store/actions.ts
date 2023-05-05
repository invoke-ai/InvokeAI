import { createAction } from '@reduxjs/toolkit';
import { InvokeTabName } from 'features/ui/store/tabMap';

export const userInvoked = createAction<InvokeTabName>('app/userInvoked');
