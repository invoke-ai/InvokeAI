/**
 * System prompts steer the Expand Prompt LLM. They are user-owned named records that may be
 * shared with everyone on the install.
 *
 * Ownership mirrors prompt templates, with one deliberate difference: there is no "default"
 * flag. The backend seeds its built-ins under the user id `system`, but that same id is what
 * *every* request carries in single-user mode, so an install that later switched to multiuser
 * has ordinary user-created prompts owned by `system` too. Visibility and editability are
 * therefore decided by `is_public` and ownership alone — never by comparing against `system`.
 */

export interface SystemPromptAccount {
  currentUserId: string | null;
  multiuserEnabled: boolean;
}

interface SystemPromptOwnershipRecord {
  isPublic: boolean;
  userId: string;
}

export interface ClassifiedSystemPrompts<T> {
  personalPrompts: T[];
  sharedPrompts: T[];
  /** The records visible to this account, personal first. */
  prompts: T[];
}

export const classifySystemPrompts = <T extends SystemPromptOwnershipRecord>(
  records: readonly T[],
  account: SystemPromptAccount
): ClassifiedSystemPrompts<T> => {
  const personalPrompts: T[] = [];
  const sharedPrompts: T[] = [];

  for (const record of records) {
    if (!account.multiuserEnabled || record.userId === account.currentUserId) {
      personalPrompts.push(record);
    } else if (record.isPublic) {
      sharedPrompts.push(record);
    }
  }

  return { personalPrompts, prompts: [...personalPrompts, ...sharedPrompts], sharedPrompts };
};

export const isOwnedSystemPrompt = (record: SystemPromptOwnershipRecord, account: SystemPromptAccount): boolean =>
  !account.multiuserEnabled || (account.currentUserId !== null && record.userId === account.currentUserId);

export class SystemPromptOwnershipError extends Error {
  constructor() {
    super('Only personal system prompts can be changed.');
    this.name = 'SystemPromptOwnershipError';
  }
}

export const requireOwnedSystemPrompt = (record: SystemPromptOwnershipRecord, account: SystemPromptAccount): void => {
  if (!isOwnedSystemPrompt(record, account)) {
    throw new SystemPromptOwnershipError();
  }
};

/**
 * The selection that should actually be in effect.
 *
 * The stored id outlives the record it points at — another tab can delete it, and a shared
 * prompt stops being visible when its owner unshares it. Rather than persist a corrective
 * write, the selection is resolved on read: a stale id falls back to the first visible prompt,
 * which is also what an install with no selection yet gets. Returns null only when there is
 * nothing to select.
 */
export const resolveSelectedSystemPromptId = (
  prompts: readonly { id: string }[],
  selectedId: string | null
): string | null => {
  if (selectedId !== null && prompts.some((prompt) => prompt.id === selectedId)) {
    return selectedId;
  }

  return prompts[0]?.id ?? null;
};
