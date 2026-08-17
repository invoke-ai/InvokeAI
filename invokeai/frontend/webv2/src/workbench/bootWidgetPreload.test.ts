import { describe, expect, it, vi } from 'vitest';

import { readBootWidgetHint, writeBootWidgetHint } from './bootWidgetPreload';
import { defaultLayoutPreset } from './layoutPresets';
import { getLayoutWidgetTypeIds } from './layoutWidgetSet';

const storage = new Map<string, string>();

vi.stubGlobal('window', {
  localStorage: {
    getItem: (key: string): string | null => storage.get(key) ?? null,
    setItem: (key: string, value: string): void => {
      storage.set(key, value);
    },
  },
});

const HINT_KEY = 'invokeai:v7:webv2:boot-widgets';

describe('boot widget preload', () => {
  it('derives the default layout boot set: active instances plus every bottom item', () => {
    const typeIds = getLayoutWidgetTypeIds(defaultLayoutPreset.snapshot);

    for (const region of Object.values(defaultLayoutPreset.snapshot.widgetRegions)) {
      const activeTypeId = defaultLayoutPreset.snapshot.widgetInstances[region.activeInstanceId]?.typeId;

      if (activeTypeId) {
        expect(typeIds).toContain(activeTypeId);
      }
    }

    // The status bar mounts all bottom items, not just the active one.
    for (const instanceId of defaultLayoutPreset.snapshot.widgetRegions.bottom.instanceIds) {
      const typeId = defaultLayoutPreset.snapshot.widgetInstances[instanceId]?.typeId;

      if (typeId) {
        expect(typeIds).toContain(typeId);
      }
    }
  });

  it('round-trips the hint through storage', () => {
    writeBootWidgetHint(['canvas', 'gallery', 'server-status']);

    expect(readBootWidgetHint()).toEqual(['canvas', 'gallery', 'server-status']);
  });

  it('rejects malformed hints instead of throwing', () => {
    storage.set(HINT_KEY, 'not json {');
    expect(readBootWidgetHint()).toBeNull();

    storage.set(HINT_KEY, '{"widgets":"canvas"}');
    expect(readBootWidgetHint()).toBeNull();

    storage.set(HINT_KEY, '[]');
    expect(readBootWidgetHint()).toBeNull();

    storage.set(HINT_KEY, '[1, "", "canvas"]');
    expect(readBootWidgetHint()).toEqual(['canvas']);
  });

  it('caps an oversized hint instead of fanning out unbounded preloads', () => {
    storage.set(HINT_KEY, JSON.stringify(Array.from({ length: 100 }, (_, index) => `widget-${index}`)));

    expect(readBootWidgetHint()).toHaveLength(32);
  });
});
