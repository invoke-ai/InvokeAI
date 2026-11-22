import { describe, expect, it } from 'vitest';

import {
  getNaturalDestination,
  getPlacedWidgetTypeIds,
  getGraphWidgetSources,
  getVisibleWidgetTypeIds,
} from './graphWidgets';
import { getSourceIdForWidgetTypeId, getWidgetTypeIdForSourceId } from './invocationSources';
import { firstPartyWidgetManifests } from './widgets/manifests';
import { createInitialWorkbenchState, getLayoutOf, workbenchReducer } from './workbenchState.testing';

const applyPreset = (presetId: string) =>
  getLayoutOf(workbenchReducer(createInitialWorkbenchState(), { presetId, type: 'applyPreset' }));

describe('graph widget sources', () => {
  it('maps widgets and sources only through manifest graphBearing', () => {
    for (const manifest of firstPartyWidgetManifests) {
      const sourceId = manifest.graphBearing?.sourceId ?? null;

      expect(getSourceIdForWidgetTypeId(manifest.id)).toBe(sourceId);
      if (sourceId) {
        expect(getWidgetTypeIdForSourceId(sourceId)).toBe(manifest.id);
      }
    }
    expect(getSourceIdForWidgetTypeId('third-party')).toBeNull();
  });

  // Keep unplaced sources selectable so switching sources does not require manual layout changes.
  it('offers every graph widget as a source', () => {
    expect(getGraphWidgetSources().map((source) => source.sourceId)).toEqual([
      'generate',
      'canvas',
      'upscale',
      'video',
      'workflow',
    ]);
  });

  it('separates active widgets from placed secondary widgets', () => {
    const project = applyPreset('edit');

    // Edit shows Generate on the left and Canvas in the centre...
    expect([...getVisibleWidgetTypeIds(project)]).toEqual(expect.arrayContaining(['generate', 'canvas']));
    expect(getVisibleWidgetTypeIds(project).has('workflow')).toBe(false);

    expect([...getPlacedWidgetTypeIds(project)]).toEqual(expect.arrayContaining(['generate', 'canvas', 'upscale']));
    expect(getPlacedWidgetTypeIds(project).has('workflow')).toBe(false);
  });

  it('counts a floated widget as both visible and placed', () => {
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'upscale', type: 'floatWidget' });
    const project = getLayoutOf(state);

    expect(project.floatingWidgets?.upscale).toBeDefined();
    expect(getVisibleWidgetTypeIds(project).has('upscale')).toBe(true);
    expect(getPlacedWidgetTypeIds(project).has('upscale')).toBe(true);
  });

  it('maps each source to its natural output target', () => {
    expect(getNaturalDestination('canvas')).toBe('canvas');
    expect(getNaturalDestination('generate')).toBe('gallery');
    expect(getNaturalDestination('workflow')).toBe('gallery');
  });
});
