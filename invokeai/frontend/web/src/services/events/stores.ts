import { round } from 'es-toolkit/compat';
import { atom, computed } from 'nanostores';
import type { S } from 'services/api/types';
import type { AppSocket } from 'services/events/types';

export const $socket = atom<AppSocket | null>(null);
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);

export const $lastProgressMessage = computed($lastProgressEvent, (val) => {
  if (!val) {
    return null;
  }
  return formatProgressMessage(val);
});
export const formatProgressMessage = (data: S['InvocationProgressEvent']): string => {
  let message = data.message;
  if (data.percentage) {
    message += ` (${round(data.percentage * 100)}%)`;
  }
  return message;
};
