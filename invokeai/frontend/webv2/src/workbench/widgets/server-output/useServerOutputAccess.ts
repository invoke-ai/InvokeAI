import { getCapabilities, useAuthSession } from '@features/identity';
import { useWidgetCapabilities } from '@workbench/widgetCapabilities';
import { useSyncExternalStore } from 'react';

import type { ServerOutputController, ServerOutputControllerSnapshot } from './controller';
import type { ServerOutputReaderSnapshot } from './reader';
import type { ServerOutputAccess } from './serverOutputStatus';

const NO_CONTROLLER: ServerOutputControllerSnapshot = Object.freeze({ attachment: null, generation: 0, refusal: null });
const noSubscription = () => () => {};
const getNoController = () => NO_CONTROLLER;
const getNoReader = () => null;

export const useServerOutputController = (
  controller: ServerOutputController | undefined
): ServerOutputControllerSnapshot =>
  useSyncExternalStore(
    controller?.subscribe ?? noSubscription,
    controller?.getSnapshot ?? getNoController,
    controller?.getSnapshot ?? getNoController
  );

/** The mounted reader's snapshot, or null while none is mounted. */
export const useServerOutputReader = (snapshot: ServerOutputControllerSnapshot): ServerOutputReaderSnapshot | null => {
  const reader = snapshot.attachment?.reader;

  return useSyncExternalStore(
    reader?.subscribe ?? noSubscription,
    reader?.getSnapshot ?? getNoReader,
    reader?.getSnapshot ?? getNoReader
  );
};

/** Also observes the frontend config, so the view loads its grant and retries a failed read when it mounts. */
export const useServerOutputAccess = (snapshot: ServerOutputControllerSnapshot): ServerOutputAccess => {
  const session = useAuthSession();
  const { viewServerOutput } = useWidgetCapabilities();

  return {
    canViewByRole: getCapabilities(session).canViewServerOutput,
    capability: viewServerOutput,
    multiuser: session.multiuserEnabled,
    refusal: snapshot.refusal,
  };
};
