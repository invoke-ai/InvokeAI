import { describe, expect, it } from 'vitest';

import {
  classifySystemPrompts,
  isOwnedSystemPrompt,
  parseMaxTokensInput,
  buildDuplicateName,
  canEditSystemPrompt,
  requireEditableSystemPrompt,
  resolveSelectedSystemPromptId,
  SYSTEM_PROMPT_MAX_TOKENS_MAX,
  SYSTEM_PROMPT_MAX_TOKENS_MIN,
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
    // The synthetic system identity also owns ordinary single-user records.
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

describe('isOwnedSystemPrompt / requireEditableSystemPrompt', () => {
  it('allows the owner and refuses another user’s shared prompt', () => {
    expect(isOwnedSystemPrompt(record('a', 'me', false), MULTIUSER)).toBe(true);
    expect(isOwnedSystemPrompt(record('b', 'someone', true), MULTIUSER)).toBe(false);

    const noAuthority = { ...MULTIUSER, canManageSharedPrompts: false };

    expect(() => requireEditableSystemPrompt(record('b', 'someone', true), noAuthority)).toThrow(
      SystemPromptOwnershipError
    );
    expect(() => requireEditableSystemPrompt(record('a', 'me', false), noAuthority)).not.toThrow();
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
    // Recover selection at read time after deletion/unsharing without corrective persistence.
    expect(resolveSelectedSystemPromptId(prompts, 'deleted')).toBe('a');
  });

  it('returns null when there is nothing to select', () => {
    expect(resolveSelectedSystemPromptId([], 'anything')).toBeNull();
    expect(resolveSelectedSystemPromptId([], null)).toBeNull();
  });
});

describe('parseMaxTokensInput', () => {
  it('reads an empty field as "use the backend default"', () => {
    expect(parseMaxTokensInput('')).toBeNull();
    expect(parseMaxTokensInput('   ')).toBeNull();
  });

  it('accepts a whole number inside the range the endpoint enforces', () => {
    expect(parseMaxTokensInput('500')).toBe(500);
    expect(parseMaxTokensInput(String(SYSTEM_PROMPT_MAX_TOKENS_MIN))).toBe(SYSTEM_PROMPT_MAX_TOKENS_MIN);
    expect(parseMaxTokensInput(String(SYSTEM_PROMPT_MAX_TOKENS_MAX))).toBe(SYSTEM_PROMPT_MAX_TOKENS_MAX);
  });

  it('rejects values the endpoint would 422 on', () => {
    expect(parseMaxTokensInput('0')).toBe('invalid');
    expect(parseMaxTokensInput(String(SYSTEM_PROMPT_MAX_TOKENS_MAX + 1))).toBe('invalid');
  });

  it('rejects anything that is not a plain integer', () => {
    // All of these survive `Number()`, and none of them is a token count someone meant to type.
    for (const raw of ['12e2', '1.5', '-5', '0x10', 'abc', '5 0']) {
      expect(parseMaxTokensInput(raw)).toBe('invalid');
    }
  });
});

describe('canEditSystemPrompt', () => {
  const MANAGER = { ...MULTIUSER, canManageSharedPrompts: true };
  const MEMBER = { ...MULTIUSER, canManageSharedPrompts: false };

  it('lets a manager edit a prompt shared by someone else', () => {
    const sharedByOther = record('b', 'someone', true);

    expect(canEditSystemPrompt(sharedByOther, MEMBER)).toBe(false);
    expect(canEditSystemPrompt(sharedByOther, MANAGER)).toBe(true);
  });

  it('never lets a manager edit another user’s private prompt', () => {
    // Private-prompt moderation rights do not grant everyday editing controls.
    const privateOfOther = record('c', 'someone', false);

    expect(canEditSystemPrompt(privateOfOther, MANAGER)).toBe(false);
    expect(() => requireEditableSystemPrompt(privateOfOther, MANAGER)).toThrow(SystemPromptOwnershipError);
  });

  it('leaves own prompts editable without any management rights', () => {
    expect(canEditSystemPrompt(record('a', 'me', false), MEMBER)).toBe(true);
  });

  it('keeps everything editable in single-user mode', () => {
    const singleUser = { ...SINGLE_USER, canManageSharedPrompts: false };

    expect(canEditSystemPrompt(record('a', 'system', true), singleUser)).toBe(true);
  });
});

describe('buildDuplicateName', () => {
  it('suffixes the source name', () => {
    expect(buildDuplicateName('Ref2VA', [])).toBe('Ref2VA (copy)');
  });

  it('numbers past names already taken, so repeated copies stay distinguishable', () => {
    expect(buildDuplicateName('Ref2VA', ['Ref2VA (copy)'])).toBe('Ref2VA (copy 2)');
    expect(buildDuplicateName('Ref2VA', ['Ref2VA (copy)', 'Ref2VA (copy 2)'])).toBe('Ref2VA (copy 3)');
  });

  it('skips a gap rather than reusing a name in the list', () => {
    expect(buildDuplicateName('Ref2VA', ['Ref2VA (copy)', 'Ref2VA (copy 3)'])).toBe('Ref2VA (copy 2)');
  });

  it('copies a copy without parsing the suffix apart', () => {
    expect(buildDuplicateName('Ref2VA (copy)', ['Ref2VA (copy)'])).toBe('Ref2VA (copy) (copy)');
  });
});
