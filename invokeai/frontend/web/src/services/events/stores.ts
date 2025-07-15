import { round } from 'es-toolkit/compat';
import type { EphemeralProgressImage } from 'features/controlLayers/store/types';
import { atom, computed, map } from 'nanostores';
import type { S } from 'services/api/types';
import type { AppSocket } from 'services/events/types';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';

export const $socket = atom<AppSocket | null>(null);
export const $socketOptions = map<Partial<ManagerOptions & SocketOptions>>({});
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);

export const $lastWorkflowsProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastWorkflowsProgressImage = atom<EphemeralProgressImage | null>(null);
export const $lastUpscalingProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastUpscalingProgressImage = atom<EphemeralProgressImage | null>(null);

export const $lastProgressImage = computed($lastProgressEvent, (val) => val?.image ?? null);
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
