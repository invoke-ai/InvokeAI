import type { Viewport } from '@xyflow/react';

import { describe, expect, it } from 'vitest';

import { getWorkflowViewport, setWorkflowViewport } from './workflowViewportStore';

describe('workflowViewportStore', () => {
  it('stores workflow editor viewports per widget instance for the current session', () => {
    const viewport: Viewport = { x: 12, y: 24, zoom: 0.75 };

    expect(getWorkflowViewport('workflow:center')).toBeNull();

    setWorkflowViewport('workflow:center', viewport);

    expect(getWorkflowViewport('workflow:center')).toEqual(viewport);
    expect(getWorkflowViewport('workflow:bottom')).toBeNull();
  });
});
