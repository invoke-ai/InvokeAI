import type { QueueItemReadModel } from '@features/queue/contracts';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type { ImageRecallKind } from '@workbench/image-actions';

import { ensureModelsLoaded, getModelsSnapshot } from '@features/models';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

import { executeQueueItemRecall, type QueueItemRecallContext } from './executeQueueItemRecall';
import { readQueueItemRecallInput } from './queueRecall';
import { loadLocalRecallSnapshot } from './useLocalRecallSnapshot';

/**
 * Recalls from an item no menu has open, such as the focused queue row on a key press: what the open menus read as
 * they mount (the snapshot, the model catalog) is read first. The caller captures the account and project at the
 * press, before loading this module, so a switch while anything loads drops the recall.
 */
export const recallQueueItem = async ({
  context,
  item,
  kind,
  owner,
  projectId,
  queryClient,
}: {
  context: QueueItemRecallContext;
  item: QueueItemReadModel;
  kind: ImageRecallKind;
  owner: AccountScope;
  projectId: string;
  queryClient: QueryClient;
}): Promise<void> => {
  if (!isAccountScopeCurrent(owner) || !context.queries.isActiveProject(projectId)) {
    return;
  }
  const [snapshot] = await Promise.all([
    loadLocalRecallSnapshot({ origin: item.origin, projects: context.queries.getSnapshot().projects, queryClient }),
    // A catalog that fails to load leaves the recall to say what it cannot apply.
    ensureModelsLoaded().catch(() => undefined),
  ]);
  if (!isAccountScopeCurrent(owner) || !context.queries.isActiveProject(projectId)) {
    return;
  }
  const { isNonGenerateRun, meta } = readQueueItemRecallInput(item);
  await executeQueueItemRecall({
    context,
    isNonGenerateRun,
    kind,
    meta,
    models: getModelsSnapshot().models,
    snapshot,
  });
};
