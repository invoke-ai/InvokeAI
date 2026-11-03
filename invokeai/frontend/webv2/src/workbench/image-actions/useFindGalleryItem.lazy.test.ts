import { describe, expect, it, vi } from 'vitest';

vi.mock('@features/gallery/queries', () => {
  throw new Error('The Gallery data module must stay behind the reveal dynamic import.');
});

vi.mock('@tanstack/react-query', () => ({ useQueryClient: vi.fn() }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: vi.fn() }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: vi.fn(),
  useWorkbenchQueries: vi.fn(),
}));

describe('useFindGalleryItem lazy boundary', () => {
  it('loads without importing Gallery data queries', async () => {
    await expect(import('./useFindGalleryItem')).resolves.toHaveProperty('useFindGalleryItem');
  });
});
