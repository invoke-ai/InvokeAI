import { describe, expect, it } from 'vitest';

import { loadWorkbenchCommandPaletteDialog } from './WorkbenchCommandPalette';

describe('Workbench command-palette lazy boundary', () => {
  it('loads the real workbench dialog module in the browser', async () => {
    const module = await loadWorkbenchCommandPaletteDialog();

    expect(module.default).toEqual(expect.any(Function));
  });
});
