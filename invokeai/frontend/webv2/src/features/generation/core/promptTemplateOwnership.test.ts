import { describe, expect, it } from 'vitest';

import {
  classifyPromptTemplates,
  PromptTemplateOwnershipError,
  requireOwnedPromptTemplate,
} from './promptTemplateOwnership';

interface Template {
  id: string;
  isDefault: boolean;
  isPublic: boolean;
  userId: string;
}

const template = (overrides: Partial<Template> = {}): Template => ({
  id: 'template',
  isDefault: false,
  isPublic: false,
  userId: 'owner',
  ...overrides,
});

describe('prompt template ownership policy', () => {
  it('classifies every non-default template as personal in single-user mode', () => {
    const records = [template({ id: 'private' }), template({ id: 'public', isPublic: true, userId: 'other' })];

    expect(
      classifyPromptTemplates(records, { currentUserId: null, multiuserEnabled: false }).personalTemplates
    ).toEqual(records);
  });

  it('separates owned, shared, built-in, and hidden records in multi-user mode', () => {
    const personal = template({ id: 'personal' });
    const shared = template({ id: 'shared', isPublic: true, userId: 'other' });
    const hidden = template({ id: 'hidden', userId: 'other' });
    const builtIn = template({ id: 'built-in', isDefault: true, userId: 'system' });

    const result = classifyPromptTemplates([personal, shared, hidden, builtIn], {
      currentUserId: 'owner',
      multiuserEnabled: true,
    });

    expect(result.personalTemplates).toEqual([personal]);
    expect(result.sharedTemplates).toEqual([shared]);
    expect(result.defaultTemplates).toEqual([builtIn]);
    expect(result.templates).toEqual([personal, shared, builtIn]);
  });

  it('exposes only built-ins while multi-user identity is unresolved', () => {
    const personal = template();
    const builtIn = template({ id: 'built-in', isDefault: true });

    expect(classifyPromptTemplates([personal, builtIn], { currentUserId: null, multiuserEnabled: true })).toEqual({
      defaultTemplates: [builtIn],
      personalTemplates: [],
      sharedTemplates: [],
      templates: [builtIn],
    });
  });

  it('does not grant a foreign private record to an administrator-shaped viewer', () => {
    const foreignPrivate = template({ userId: 'other' });

    expect(
      classifyPromptTemplates([foreignPrivate], {
        currentUserId: 'admin',
        multiuserEnabled: true,
      }).templates
    ).toEqual([]);
  });

  it('rejects mutations of shared and built-in records', () => {
    const account = { currentUserId: 'owner', multiuserEnabled: true };

    expect(() => requireOwnedPromptTemplate(template({ isPublic: true, userId: 'other' }), account)).toThrow(
      PromptTemplateOwnershipError
    );
    expect(() => requireOwnedPromptTemplate(template({ isDefault: true }), account)).toThrow(
      PromptTemplateOwnershipError
    );
    expect(() => requireOwnedPromptTemplate(template(), account)).not.toThrow();
  });
});
