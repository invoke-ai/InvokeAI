import { atom, computed, map } from 'nanostores';
import type { S } from 'services/api/types';
import type { AppSocket } from 'services/events/types';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';

export const $socket = atom<AppSocket | null>(null);
export const $socketOptions = map<Partial<ManagerOptions & SocketOptions>>({});
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $progressImage = computed($lastProgressEvent, (val) => val?.image ?? null);
export const $hasProgressImage = computed($lastProgressEvent, (val) => Boolean(val?.image));
export const $isProgressFromCanvas = computed($lastProgressEvent, (val) => val?.destination === 'canvas');
