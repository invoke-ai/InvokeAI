import type { MouseEvent } from 'react';

export const preventDefault = (e: MouseEvent) => {
  e.preventDefault();
};
