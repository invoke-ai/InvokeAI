import { describe, expect, it } from 'vitest';

import { createDeterministicProjectId } from './ids';

describe('project ids', () => {
  it('derives stable, distinct UUID-shaped ids from idempotency scopes', async () => {
    const first = await createDeterministicProjectId('project-1\u0000revision:4');

    await expect(createDeterministicProjectId('project-1\u0000revision:4')).resolves.toBe(first);
    await expect(createDeterministicProjectId('project-1\u0000revision:5')).resolves.not.toBe(first);
    expect(first).toMatch(/^project-[0-9a-f]{8}-[0-9a-f]{4}-8[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });
});
