import { createAction } from '@reduxjs/toolkit';

export const videoModalLinkClicked = createAction<string>('system/videoModalLinkClicked');
export const videoModalOpened = createAction('system/videoModalOpened');
