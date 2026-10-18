import { describe, expect, it, vi } from 'vitest';

vi.mock('./revealExternalGeneration', () => {
  throw new Error('The external generation reveal must stay behind its dynamic import.');
});
vi.mock('./revealGalleryItem', () => {
  throw new Error('Gallery reveal must stay behind the external reveal dynamic import.');
});
vi.mock('@features/queue/queries', () => {
  throw new Error('Queue result reads must stay behind the external reveal dynamic import.');
});

describe('recall and reveal bridge lazy boundary', () => {
  it('loads in the editor boot graph without the reveal it attaches', async () => {
    await expect(import('./recallParametersBridge')).resolves.toHaveProperty('attachExternalGenerationReveal');
  });
});
