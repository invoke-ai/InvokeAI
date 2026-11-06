import { afterEach, describe, expect, it } from 'vitest';

import {
  getWorkflowViewCenter,
  registerWorkflowFlowInstance,
  registerWorkflowViewElement,
  releaseWorkflowFlowInstance,
  type WorkflowFlowInstance,
} from './flowInstanceStore';

// Screen and graph coordinates differ by a pan of (-10, -20) at zoom 2, as a moved editor would.
const flowInstance = {
  screenToFlowPosition: ({ x, y }: { x: number; y: number }) => ({ x: (x + 10) / 2, y: (y + 20) / 2 }),
} as unknown as WorkflowFlowInstance;

describe('getWorkflowViewCenter', () => {
  afterEach(() => {
    releaseWorkflowFlowInstance(flowInstance);
    document.body.replaceChildren();
  });

  it('places at the middle of the editor frame, not of the browser window, and forgets a frame that left', () => {
    const frame = document.createElement('div');

    frame.style.cssText = 'position: fixed; left: 100px; top: 50px; width: 400px; height: 200px;';
    document.body.append(frame);
    registerWorkflowFlowInstance(flowInstance);

    expect(getWorkflowViewCenter()).toBeNull();

    const release = registerWorkflowViewElement(frame);

    expect(getWorkflowViewCenter()).toEqual({ x: 155, y: 85 });

    release?.();

    expect(getWorkflowViewCenter()).toBeNull();
  });
});
