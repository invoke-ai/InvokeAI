import { describe, expect, it } from 'vitest';

import {
  describeRoute,
  getNaturalDestination,
  getPlacedWidgetTypeIds,
  getSourceIdForWidgetTypeId,
  getVisibleWidgetTypeIds,
  getWidgetTypeIdForSourceId,
  graphWidgetSources,
  isGraphWidgetTypeId,
} from './graphWidgets';
import { createInitialWorkbenchState, workbenchReducer } from './workbenchState.testing';

const applyPreset = (presetId: string) => {
  const state = workbenchReducer(createInitialWorkbenchState(), { presetId, type: 'applyPreset' });

  return state.projects.find((candidate) => candidate.id === state.activeProjectId)!;
};

describe('graph widget sources', () => {
  it('recognises only the graph-bearing widget types', () => {
    expect(isGraphWidgetTypeId('generate')).toBe(true);
    expect(isGraphWidgetTypeId('canvas')).toBe(true);
    expect(isGraphWidgetTypeId('upscale')).toBe(true);
    expect(isGraphWidgetTypeId('workflow')).toBe(true);
    expect(isGraphWidgetTypeId('gallery')).toBe(false);
    expect(isGraphWidgetTypeId('preview')).toBe(false);
    expect(getSourceIdForWidgetTypeId('gallery')).toBeNull();
    expect(getSourceIdForWidgetTypeId('workflow')).toBe('workflow');
    expect(getWidgetTypeIdForSourceId('upscale')).toBe('upscale');
  });

  // Every source stays pickable. Hiding the ones the layout does not happen to
  // be showing turns the menu into a dead end: the user can see that Workflow
  // exists but has no way to route to it without rearranging the dock first.
  it('offers every graph widget as a source', () => {
    expect(graphWidgetSources.map((source) => source.sourceId)).toEqual(['generate', 'canvas', 'upscale', 'workflow']);
  });

  it('separates active widgets from placed secondary widgets', () => {
    const project = applyPreset('edit');

    // Edit shows Generate on the left and Canvas in the centre...
    expect([...getVisibleWidgetTypeIds(project)]).toEqual(expect.arrayContaining(['generate', 'canvas']));
    expect(getVisibleWidgetTypeIds(project).has('workflow')).toBe(false);

    // ...while Upscale is still placed second in the left rail, so routing to
    // it needs no new placement, only revealing. Workflow belongs to Automate.
    expect([...getPlacedWidgetTypeIds(project)]).toEqual(expect.arrayContaining(['generate', 'canvas', 'upscale']));
    expect(getPlacedWidgetTypeIds(project).has('workflow')).toBe(false);
  });

  it('counts a floated widget as both visible and placed', () => {
    // A widget in a window is on screen; reading only the rails would drop the
    // first graph-bearing widget that floats out of the invoke-source list.
    let state = workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' });
    state = workbenchReducer(state, { instanceId: 'upscale', type: 'floatWidget' });
    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId)!;

    expect(project.floatingWidgets?.upscale).toBeDefined();
    expect(getVisibleWidgetTypeIds(project).has('upscale')).toBe(true);
    expect(getPlacedWidgetTypeIds(project).has('upscale')).toBe(true);
  });

  it('maps each source to its natural output target', () => {
    expect(getNaturalDestination('canvas')).toBe('canvas');
    expect(getNaturalDestination('generate')).toBe('gallery');
    expect(getNaturalDestination('workflow')).toBe('gallery');
  });

  // The indicator is icon-only, so the whole route — including whether it is
  // pinned — has to live in its accessible name.
  it('describes the whole route for assistive technology', () => {
    expect(
      describeRoute({
        destination: 'gallery',
        destinationLocked: false,
        hasSource: true,
        sourceId: 'generate',
        sourceLocked: false,
      })
    ).toBe('Invoke from generate, output to gallery, following edits');

    expect(
      describeRoute({
        destination: 'canvas',
        destinationLocked: true,
        hasSource: true,
        sourceId: 'workflow',
        sourceLocked: true,
      })
    ).toBe('Invoke from workflow, output to canvas, source locked, destination locked');

    expect(
      describeRoute({
        destination: 'gallery',
        destinationLocked: false,
        hasSource: false,
        sourceId: 'generate',
        sourceLocked: false,
      })
    ).toBe('No source widget open, output to gallery, following edits');
  });
});
