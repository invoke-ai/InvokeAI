type CompletedInvocationKeysByItemId = Map<number, Set<string>>;

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
