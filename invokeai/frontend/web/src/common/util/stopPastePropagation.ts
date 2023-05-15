import { ClipboardEvent } from 'react';

export const stopPastePropagation = (e: ClipboardEvent) => {
  e.stopPropagation();
};
