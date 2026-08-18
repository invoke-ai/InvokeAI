import { describe, expect, it } from 'vitest';

import { layoutPresets } from './layoutPresets';
import { getLayoutWidgetTypeIds } from './layoutWidgetSet';

describe('layout widget set', () => {
  it('includes every bottom instance, not just the active one', () => {
    for (const preset of layoutPresets) {
      const typeIds = getLayoutWidgetTypeIds(preset.snapshot);

      for (const instanceId of preset.snapshot.widgetRegions.bottom.instanceIds) {
        const typeId = preset.snapshot.widgetInstances[instanceId]?.typeId;

        if (typeId) {
          expect(typeIds).toContain(typeId);
        }
      }
    }
  });

  it('orders region actives ahead of the rest of the bottom strip', () => {
    const preset = layoutPresets.find(({ id }) => id === 'edit')!;
    const typeIds = getLayoutWidgetTypeIds(preset.snapshot);
    const centerTypeId = preset.snapshot.widgetInstances[preset.snapshot.widgetRegions.center.activeInstanceId]!.typeId;

    expect(typeIds[0]).toBe(centerTypeId);
    expect(typeIds.indexOf('server-status')).toBeGreaterThan(typeIds.indexOf(centerTypeId));
  });

  it('deduplicates a type placed in more than one region', () => {
    const preset = layoutPresets.find(({ id }) => id === 'compose')!;
    const typeIds = getLayoutWidgetTypeIds(preset.snapshot);

    expect(new Set(typeIds).size).toBe(typeIds.length);
  });

  it('skips instance ids that have no instance', () => {
    const typeIds = getLayoutWidgetTypeIds({
      widgetInstances: { real: { typeId: 'gallery' } },
      widgetRegions: {
        bottom: { activeInstanceId: 'missing', instanceIds: ['missing', 'real'] },
        center: { activeInstanceId: 'missing', instanceIds: ['missing'] },
      },
    });

    expect(typeIds).toEqual(['gallery']);
  });
});
