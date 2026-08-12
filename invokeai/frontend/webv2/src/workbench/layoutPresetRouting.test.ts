import { describe, expect, it } from 'vitest';

import { getLayoutPresetSourceOptions } from './layoutPresetRouting';
import { layoutPresets } from './layoutPresets';

describe('layout preset routing', () => {
  it('offers only graph sources contained in the saved arrangement', () => {
    const compose = layoutPresets[0];
    const generateOnly = {
      ...compose,
      snapshot: {
        ...compose.snapshot,
        widgetRegions: {
          ...compose.snapshot.widgetRegions,
          bottom: {
            ...compose.snapshot.widgetRegions.bottom,
            instanceIds: compose.snapshot.widgetRegions.bottom.instanceIds.filter(
              (instanceId) => instanceId !== 'workflow:bottom'
            ),
          },
          center: { ...compose.snapshot.widgetRegions.center, activeInstanceId: 'preview', instanceIds: ['preview'] },
          left: { ...compose.snapshot.widgetRegions.left, activeInstanceId: 'generate', instanceIds: ['generate'] },
          right: { ...compose.snapshot.widgetRegions.right, activeInstanceId: 'gallery', instanceIds: ['gallery'] },
        },
      },
    };

    expect(getLayoutPresetSourceOptions(generateOnly)).toEqual([
      { label: 'Generate', sourceId: 'generate', typeId: 'generate' },
    ]);
  });
});
