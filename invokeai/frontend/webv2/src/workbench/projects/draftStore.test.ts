import { describe, expect, it } from 'vitest';

import type { ProjectDraftInput } from './draftStore';

import { createMemoryProjectDraftStore, getUtf8ByteSize } from './draftStore';
import { createProjectDraftInput, testProjectDraftStoreContract } from './draftStore.contract';

describe('memory project draft store contract', () => {
  testProjectDraftStoreContract(() => Promise.resolve(createMemoryProjectDraftStore()));

  it('derives the exact UTF-8 byte count instead of trusting caller metadata', async () => {
    const store = createMemoryProjectDraftStore({ maxDraftBytes: 4 });

    await expect(
      store.stage({ ...createProjectDraftInput({ documentJson: 'éé' }), documentByteSize: 999 } as ProjectDraftInput)
    ).resolves.toEqual({
      kind: 'stored',
    });
    await expect(
      store.stage({
        ...createProjectDraftInput({ documentJson: 'ééx', generation: 2 }),
        documentByteSize: 1,
      } as ProjectDraftInput)
    ).resolves.toEqual({ kind: 'too-large' });
    await expect(store.get('project-1', 'session-a')).resolves.toMatchObject({
      draft: { documentByteSize: 4 },
      kind: 'found',
    });
  });

  it.each(['plain ascii', 'café', '😀', '\ud800', 'a😀é\udfff'])(
    'counts UTF-8 bytes like TextEncoder for %j',
    (value) => {
      expect(getUtf8ByteSize(value)).toBe(new TextEncoder().encode(value).byteLength);
    }
  );
});
