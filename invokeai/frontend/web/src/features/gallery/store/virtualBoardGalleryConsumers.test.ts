import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8');

describe('virtual board gallery consumers', () => {
  it.each([
    ['range selection', './selectCachedGalleryItemNames.ts'],
    ['board auto-selection', '../../../app/store/middleware/listenerMiddleware/listeners/boardIdSelected.ts'],
    ['initial board auto-selection', '../../../app/store/middleware/listenerMiddleware/listeners/appStarted.ts'],
  ])('%s reads the virtual-board item-name cache', (_label, relativePath) => {
    const source = readSource(relativePath);

    expect(source).toContain('getVirtualBoardItemNamesByDate');
  });
});
