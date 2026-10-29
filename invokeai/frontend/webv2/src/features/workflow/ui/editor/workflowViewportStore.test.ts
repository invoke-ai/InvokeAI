import type { Viewport } from '@xyflow/react';

import { beforeEach, describe, expect, it } from 'vitest';

import {
  clearWorkflowViewports,
  getWorkflowViewport,
  getWorkflowViewportKey,
  releaseWorkflowViewportsExcept,
  setWorkflowViewport,
} from './workflowViewportStore';

describe('workflowViewportStore', () => {
  beforeEach(() => {
    clearWorkflowViewports();
  });

  it('stores workflow editor viewports per widget instance for the current session', () => {
    const viewport: Viewport = { x: 12, y: 24, zoom: 0.75 };

    expect(getWorkflowViewport('workflow:center')).toBeNull();

    setWorkflowViewport('workflow:center', viewport);

    expect(getWorkflowViewport('workflow:center')).toEqual(viewport);
    expect(getWorkflowViewport('workflow:bottom')).toBeNull();
  });

  it('keeps one viewport per project workflow so switching brings each view back', () => {
    const first = getWorkflowViewportKey('project-1', 'wf-a', 'center');
    const second = getWorkflowViewportKey('project-1', 'wf-b', 'center');

    setWorkflowViewport(first, { x: 1, y: 1, zoom: 1 });
    setWorkflowViewport(second, { x: 2, y: 2, zoom: 2 });

    expect(first).not.toBe(second);
    expect(getWorkflowViewport(first)).toEqual({ x: 1, y: 1, zoom: 1 });
    expect(getWorkflowViewport(second)).toEqual({ x: 2, y: 2, zoom: 2 });
  });

  it('releases the viewports of workflows that left a project and leaves other projects alone', () => {
    const gone = getWorkflowViewportKey('project-1', 'wf-gone', 'center');
    const kept = getWorkflowViewportKey('project-1', 'wf-kept', 'center');
    const other = getWorkflowViewportKey('project-2', 'wf-gone', 'center');

    for (const key of [gone, kept, other]) {
      setWorkflowViewport(key, { x: 0, y: 0, zoom: 1 });
    }

    releaseWorkflowViewportsExcept('project-1', ['wf-kept']);

    expect(getWorkflowViewport(gone)).toBeNull();
    expect(getWorkflowViewport(kept)).not.toBeNull();
    expect(getWorkflowViewport(other)).not.toBeNull();
  });
});
