import { describe, expect, it } from 'vitest';

import {
  classifySystemPrompts,
  isOwnedSystemPrompt,
  requireOwnedSystemPrompt,
  resolveSelectedSystemPromptId,
  SystemPromptOwnershipError,
} from './systemPrompts';

const record = (id: string, userId: string, isPublic: boolean) => ({ id, isPublic, userId });

const MULTIUSER = { currentUserId: 'me', multiuserEnabled: true };
const SINGLE_USER = { currentUserId: 'system', multiuserEnabled: false };

describe('classifySystemPrompts', () => {
  it('splits the account owner’s prompts from prompts shared by others', () => {
    const mine = record('a', 'me', false);
    const sharedByOther = record('b', 'someone', true);
    const privateOfOther = record('c', 'someone', false);

    const classified = classifySystemPrompts([mine, sharedByOther, privateOfOther], MULTIUSER);

    expect(classified.personalPrompts).toEqual([mine]);
    expect(classified.sharedPrompts).toEqual([sharedByOther]);
    // Another user's private prompt is not visible at all.
    expect(classified.prompts).toEqual([mine, sharedByOther]);
  });

  it('treats everything as personal in single-user mode', () => {
    // The backend seeds built-ins under the user id `system`, which is also the synthetic id
    // every single-user request carries — so ownership must not be decided by comparing to it.
    const seeded = record('a', 'system', true);
    const created = record('b', 'system', false);

    const classified = classifySystemPrompts([seeded, created], SINGLE_USER);

    expect(classified.personalPrompts).toEqual([seeded, created]);
    expect(classified.sharedPrompts).toEqual([]);
  });

  it('orders personal prompts ahead of shared ones', () => {
    const sharedByOther = record('b', 'someone', true);
    const mine = record('a', 'me', false);

    expect(classifySystemPrompts([sharedByOther, mine], MULTIUSER).prompts).toEqual([mine, sharedByOther]);
  });
});

describe('isOwnedSystemPrompt / requireOwnedSystemPrompt', () => {
  it('allows the owner and refuses another user’s shared prompt', () => {
    expect(isOwnedSystemPrompt(record('a', 'me', false), MULTIUSER)).toBe(true);
    expect(isOwnedSystemPrompt(record('b', 'someone', true), MULTIUSER)).toBe(false);

    expect(() => requireOwnedSystemPrompt(record('b', 'someone', true), MULTIUSER)).toThrow(SystemPromptOwnershipError);
    expect(() => requireOwnedSystemPrompt(record('a', 'me', false), MULTIUSER)).not.toThrow();
  });

  it('treats an anonymous multiuser session as owning nothing', () => {
    const account = { currentUserId: null, multiuserEnabled: true };

    expect(isOwnedSystemPrompt(record('a', 'system', true), account)).toBe(false);
  });
});

describe('resolveSelectedSystemPromptId', () => {
  const prompts = [record('a', 'me', false), record('b', 'me', false)];

  it('keeps a selection that still exists', () => {
    expect(resolveSelectedSystemPromptId(prompts, 'b')).toBe('b');
  });

  it('falls back to the first prompt when nothing is selected yet', () => {
    expect(resolveSelectedSystemPromptId(prompts, null)).toBe('a');
  });

  it('falls back when the stored id no longer exists', () => {
    // Deleted in another tab, or a shared prompt whose owner unshared it. Resolving on read
    // means no corrective write is needed to recover.
    expect(resolveSelectedSystemPromptId(prompts, 'deleted')).toBe('a');
  });

  it('returns null when there is nothing to select', () => {
    expect(resolveSelectedSystemPromptId([], 'anything')).toBeNull();
    expect(resolveSelectedSystemPromptId([], null)).toBeNull();
  });
});
