export interface PromptTemplateAccount {
  currentUserId: string | null;
  multiuserEnabled: boolean;
}

interface PromptTemplateOwnershipRecord {
  isDefault: boolean;
  isPublic: boolean;
  userId: string;
}

export interface ClassifiedPromptTemplates<T> {
  defaultTemplates: T[];
  personalTemplates: T[];
  sharedTemplates: T[];
  /** The records visible to this account, excluding foreign private templates. */
  templates: T[];
}

export const classifyPromptTemplates = <T extends PromptTemplateOwnershipRecord>(
  records: readonly T[],
  account: PromptTemplateAccount
): ClassifiedPromptTemplates<T> => {
  const defaultTemplates: T[] = [];
  const personalTemplates: T[] = [];
  const sharedTemplates: T[] = [];

  for (const record of records) {
    if (record.isDefault) {
      defaultTemplates.push(record);
    } else if (!account.multiuserEnabled || record.userId === account.currentUserId) {
      personalTemplates.push(record);
    } else if (account.currentUserId !== null && record.isPublic) {
      sharedTemplates.push(record);
    }
  }

  return {
    defaultTemplates,
    personalTemplates,
    sharedTemplates,
    templates: [...personalTemplates, ...sharedTemplates, ...defaultTemplates],
  };
};

export class PromptTemplateOwnershipError extends Error {
  constructor() {
    super('Only personal prompt templates can be changed.');
    this.name = 'PromptTemplateOwnershipError';
  }
}

export const requireOwnedPromptTemplate = (
  record: PromptTemplateOwnershipRecord,
  account: PromptTemplateAccount
): void => {
  const isOwned =
    !record.isDefault &&
    (!account.multiuserEnabled || (account.currentUserId !== null && record.userId === account.currentUserId));

  if (!isOwned) {
    throw new PromptTemplateOwnershipError();
  }
};
