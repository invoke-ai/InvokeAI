/** The system ID is ambiguous across installation modes; use visibility and ownership to identify editable records. */

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

/** What the viewer is allowed to change, beyond what they own. */
export interface SystemPromptEditAuthority extends SystemPromptAccount {
  /** Admin-derived. Grants edit rights over shared prompts only -- see `canEditSystemPrompt`. */
  canManageSharedPrompts: boolean;
}

/** The client forbids editing another user's private prompt, including for admins. */
export const canEditSystemPrompt = (
  record: SystemPromptOwnershipRecord,
  authority: SystemPromptEditAuthority
): boolean => isOwnedSystemPrompt(record, authority) || (authority.canManageSharedPrompts && record.isPublic);

export class SystemPromptOwnershipError extends Error {
  constructor() {
    super('Only your own system prompts, or shared ones you manage, can be changed.');
    this.name = 'SystemPromptOwnershipError';
  }
}

export const requireEditableSystemPrompt = (
  record: SystemPromptOwnershipRecord,
  authority: SystemPromptEditAuthority
): void => {
  if (!canEditSystemPrompt(record, authority)) {
    throw new SystemPromptOwnershipError();
  }
};

/** Suffix collisions while preserving an existing copy suffix. */
export const buildDuplicateName = (sourceName: string, existingNames: Iterable<string>): string => {
  const taken = new Set(existingNames);
  const base = `${sourceName} (copy)`;

  if (!taken.has(base)) {
    return base;
  }
  for (let suffix = 2; ; suffix++) {
    const candidate = `${sourceName} (copy ${String(suffix)})`;

    if (!taken.has(candidate)) {
      return candidate;
    }
  }
};

/** Fall back to the first visible prompt without corrective writes; return null only for an empty list. */
export const resolveSelectedSystemPromptId = (
  prompts: readonly { id: string }[],
  selectedId: string | null
): string | null => {
  if (selectedId !== null && prompts.some((prompt) => prompt.id === selectedId)) {
    return selectedId;
  }

  return prompts[0]?.id ?? null;
};

/** Token bounds match the expansion endpoint. */
export const SYSTEM_PROMPT_MAX_TOKENS_MIN = 1;
export const SYSTEM_PROMPT_MAX_TOKENS_MAX = 2048;
/** What the backend uses when a prompt names no cap of its own. Shown as the field's placeholder. */
export const SYSTEM_PROMPT_MAX_TOKENS_DEFAULT = 300;

/** `null` = leave it to the backend default; `'invalid'` = not saveable. */
export type ParsedMaxTokens = number | null | 'invalid';

/** Keep raw drafts until save so half-typed input survives. */
export const parseMaxTokensInput = (raw: string): ParsedMaxTokens => {
  const trimmed = raw.trim();

  if (trimmed === '') {
    return null;
  }

  // Accept decimal integers only.
  if (!/^\d+$/.test(trimmed)) {
    return 'invalid';
  }

  const parsed = Number(trimmed);

  if (parsed < SYSTEM_PROMPT_MAX_TOKENS_MIN || parsed > SYSTEM_PROMPT_MAX_TOKENS_MAX) {
    return 'invalid';
  }

  return parsed;
};
