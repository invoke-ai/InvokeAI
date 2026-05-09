type CompletedInvocationKeysByItemId = Map<number, Set<string>>;

type FinishedQueueItemIds = {
  has: (itemId: number) => boolean;
};

type FinishedQueueItemInvocationEventName = 'invocation_error' | 'invocation_progress' | 'invocation_started';

export const hasCompletedInvocationKey = (
  completedInvocationKeysByItemId: CompletedInvocationKeysByItemId,
  itemId: number,
  invocationId: string
) => completedInvocationKeysByItemId.get(itemId)?.has(invocationId) ?? false;

export const markInvocationAsCompleted = (
  completedInvocationKeysByItemId: CompletedInvocationKeysByItemId,
  itemId: number,
  invocationId: string
) => {
  let completedInvocationKeys = completedInvocationKeysByItemId.get(itemId);
  if (!completedInvocationKeys) {
    completedInvocationKeys = new Set<string>();
    completedInvocationKeysByItemId.set(itemId, completedInvocationKeys);
  }
  completedInvocationKeys.add(invocationId);
};

export const clearCompletedInvocationKeysForQueueItem = (
  completedInvocationKeysByItemId: CompletedInvocationKeysByItemId,
  itemId: number
) => {
  completedInvocationKeysByItemId.delete(itemId);
};

export const shouldIgnoreFinishedQueueItemInvocationEvent = (
  eventName: FinishedQueueItemInvocationEventName,
  finishedQueueItemIds: FinishedQueueItemIds,
  itemId: number
) => {
  if (eventName === 'invocation_error') {
    return false;
  }

  return finishedQueueItemIds.has(itemId);
};
