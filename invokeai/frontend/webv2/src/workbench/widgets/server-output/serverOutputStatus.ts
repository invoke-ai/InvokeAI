import type { WidgetCapabilityState } from '@workbench/widgetContracts';

import type { ServerOutputRefusal } from './controller';
import type { ServerOutputReaderPhase } from './reader';

/** `denied`: the config refuses an admin session without saying why (capture off, or the role changed elsewhere). */
export type ServerOutputRefusalMessage = 'denied' | 'forbidden' | 'unavailable';

export type ServerOutputStatus = 'checking' | 'connecting' | 'live' | 'reconnecting' | 'unavailable';

export interface ServerOutputAccess {
  /** Identity's role check: single-user mode, or a multi-user admin. */
  readonly canViewByRole: boolean;
  readonly multiuser: boolean;
  readonly capability: WidgetCapabilityState;
  /** Why a read was refused, which outranks the config's later answer so the message stays put. */
  readonly refusal: ServerOutputRefusal | null;
}

/** Why the view refuses, or null while it may read. The view and its header both answer from this. */
export const getServerOutputRefusal = ({
  canViewByRole,
  capability,
  multiuser,
  refusal,
}: ServerOutputAccess): ServerOutputRefusalMessage | null => {
  if (refusal) {
    return refusal.reason;
  }

  if (capability !== 'denied') {
    return null;
  }

  return !canViewByRole ? 'forbidden' : multiuser ? 'denied' : 'unavailable';
};

/** The header's one word for the view's state; `phase` is the mounted reader's, null while none is mounted. */
export const getServerOutputStatus = (
  access: ServerOutputAccess,
  phase: ServerOutputReaderPhase | null
): ServerOutputStatus => {
  if (getServerOutputRefusal(access) !== null || phase === 'refused' || phase === 'unavailable') {
    return 'unavailable';
  }

  if (access.capability === 'pending') {
    return 'checking';
  }

  return phase === 'live' || phase === 'reconnecting' ? phase : 'connecting';
};
