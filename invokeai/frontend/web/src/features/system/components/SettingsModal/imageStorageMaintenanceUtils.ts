import type { S } from 'services/api/types';

export const isTerminalImageMoveJobState = (state: S['ImageMoveJobResponse']['state'] | undefined): boolean =>
  state === 'committed' || state === 'error';
