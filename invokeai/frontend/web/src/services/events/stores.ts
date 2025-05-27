import type { ProgressImage } from 'features/nodes/types/common';
import { round } from 'lodash-es';
import { atom, computed, map } from 'nanostores';
import type { S } from 'services/api/types';
import type { AppSocket } from 'services/events/types';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';

export const $socket = atom<AppSocket | null>(null);
export const $socketOptions = map<Partial<ManagerOptions & SocketOptions>>({});
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
$lastProgressEvent.subscribe((event) => {
  if (!event) {
    return;
  }
  switch (event.destination) {
    case 'workflows':
      $lastWorkflowsProgressEvent.set(event);
      if (event.image) {
        $lastWorkflowsProgressImage.set({ sessionId: event.session_id, image: event.image });
      }
      break;
    case 'upscaling':
      $lastUpscalingProgressEvent.set(event);
      if (event.image) {
        $lastUpscalingProgressImage.set({ sessionId: event.session_id, image: event.image });
      }
      break;
    case 'canvas':
      $lastCanvasProgressEvent.set(event);
      if (event.image) {
        $lastCanvasProgressImage.set({ sessionId: event.session_id, image: event.image });
      }
      break;
  }
});

type EphemeralProgressImage = { sessionId: string; image: ProgressImage };

export const $lastCanvasProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastCanvasProgressImage = atom<EphemeralProgressImage | null>(null);
export const $lastWorkflowsProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastWorkflowsProgressImage = atom<EphemeralProgressImage | null>(null);
export const $lastUpscalingProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastUpscalingProgressImage = atom<EphemeralProgressImage | null>(null);

export const $progressImage = computed($lastProgressEvent, (val) => val?.image ?? null);
export const $hasProgressImage = computed($lastProgressEvent, (val) => Boolean(val?.image));
export const $invocationProgressMessage = computed($lastProgressEvent, (val) => {
  if (!val) {
    return null;
  }

  let message = val.message;
  if (val.percentage) {
    message += ` (${round(val.percentage * 100)}%)`;
  }
  return message;
});
