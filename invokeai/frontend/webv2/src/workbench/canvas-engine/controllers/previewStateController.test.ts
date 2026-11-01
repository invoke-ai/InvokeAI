import { describe, expect, it } from 'vitest';

import { PreviewStateController } from './previewStateController';

const rect = (x: number) => ({ height: 10, width: 10, x, y: 0 });

describe('staged regions', () => {
  it('keeps each rect once, selected when any result there is, with the selected one last', () => {
    const previews = new PreviewStateController();

    expect(
      previews.setStagedRegions([
        { rect: rect(0), selected: false },
        { rect: rect(0), selected: true },
        { rect: rect(20), selected: false },
        { rect: rect(20), selected: false },
      ])
    ).toBe(true);
    expect(previews.getStagedRegions()).toEqual([
      { rect: rect(20), selected: false },
      { rect: rect(0), selected: true },
    ]);
  });

  it('reports an equal list as unchanged, and owns copies of the rects it keeps', () => {
    const previews = new PreviewStateController();
    const moving = rect(0);
    previews.setStagedRegions([{ rect: moving, selected: true }]);

    expect(previews.setStagedRegions([{ rect: rect(0), selected: true }])).toBe(false);
    moving.x = 99;
    expect(previews.getStagedRegions()).toEqual([{ rect: rect(0), selected: true }]);
    expect(previews.setStagedRegions([{ rect: rect(0), selected: false }])).toBe(true);
    expect(previews.setStagedRegions([])).toBe(true);
    expect(previews.getStagedRegions()).toEqual([]);
  });
});
