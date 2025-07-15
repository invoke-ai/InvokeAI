import { createAction } from '@reduxjs/toolkit';

export const videoModalLinkClicked = createAction<string>('system/videoModalLinkClicked');
export const videoModalOpened = createAction('system/videoModalOpened');

export const trackErrorDetails = createAction<{
  title: string;
  errorMessage?: string;
  description: string | null;
}>('system/trackErrorDetails');
