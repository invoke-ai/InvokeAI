import type { WidgetCapabilities, WidgetCapabilityState } from '@workbench/widgetContracts';

import { getAuthSession, getCapabilities, useAuthSession, type AuthSession } from '@features/identity';
import { queryClient } from '@platform/query/client';
import { useQuery, type QueryState } from '@tanstack/react-query';

import { frontendConfigQueryOptions, type FrontendConfig } from './frontendConfig';

const SNAPSHOTS: Readonly<Record<WidgetCapabilityState, WidgetCapabilities>> = {
  denied: Object.freeze({ viewServerOutput: 'denied' }),
  granted: Object.freeze({ viewServerOutput: 'granted' }),
  pending: Object.freeze({ viewServerOutput: 'pending' }),
};

type ConfigState = Pick<QueryState<FrontendConfig>, 'data' | 'fetchStatus' | 'status'>;

/**
 * Both sides must grant: the session's role (it changes at once when this tab learns of it) and the server's frontend
 * config for this account. Unanswered, or retrying after a failure, is pending; a failed or older config denies.
 */
export const resolveWidgetCapabilities = (
  session: AuthSession,
  config: ConfigState | undefined
): WidgetCapabilities => {
  if (!getCapabilities(session).canViewServerOutput) {
    return SNAPSHOTS.denied;
  }

  if (config?.data !== undefined) {
    return config.data.can_view_server_output === true ? SNAPSHOTS.granted : SNAPSHOTS.denied;
  }

  return config?.status === 'error' && config.fetchStatus === 'idle' ? SNAPSHOTS.denied : SNAPSHOTS.pending;
};

/** The grants as the app query cache holds them now, for commands and other non-render reads. */
export const readWidgetCapabilities = (): WidgetCapabilities => {
  const session = getAuthSession();

  return resolveWidgetCapabilities(
    session,
    queryClient.getQueryState<FrontendConfig>(frontendConfigQueryOptions(session).queryKey)
  );
};

/**
 * Observes the account's frontend config, so whichever view needs a grant loads it: only while the session's role
 * could be granted at all. A new observer retries a failed read, so mounting a view is its retry. Uses the app query
 * client, which `readWidgetCapabilities` reads, whether or not a provider is above. Snapshots are shared constants.
 */
export const useWidgetCapabilities = (): WidgetCapabilities => {
  const session = useAuthSession();
  const canBeGranted = getCapabilities(session).canViewServerOutput;
  const { data, fetchStatus, status } = useQuery(
    { ...frontendConfigQueryOptions(session), enabled: canBeGranted },
    queryClient
  );

  return resolveWidgetCapabilities(session, canBeGranted ? { data, fetchStatus, status } : undefined);
};
