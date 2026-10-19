import type { CanvasEngine } from '@workbench/canvas-engine/api';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const registry = vi.hoisted(() => ({
  acquire: vi.fn(),
  release: vi.fn(),
}));

vi.mock('./engineRegistry', () => ({
  getOrCreateEngine: registry.acquire,
  releaseEngine: registry.release,
}));

import { createCanvasEngineResource } from './useCanvasEngine';

describe('Canvas engine React resource', () => {
  beforeEach(() => {
    registry.acquire.mockReset();
    registry.release.mockReset();
  });

  it('acquires only while subscribed and balances one lease across subscribers', () => {
    const engine = { projectId: 'project-a' } as CanvasEngine;
    registry.acquire.mockReturnValue(engine);
    const resource = createCanvasEngineResource('project-a', {} as never);
    const firstListener = vi.fn();
    const secondListener = vi.fn();

    expect(resource.getSnapshot()).toBeNull();
    expect(registry.acquire).not.toHaveBeenCalled();

    const unsubscribeFirst = resource.subscribe(firstListener);
    expect(resource.getSnapshot()).toBe(engine);
    expect(registry.acquire).toHaveBeenCalledOnce();
    expect(firstListener).toHaveBeenCalledOnce();

    const unsubscribeSecond = resource.subscribe(secondListener);
    expect(registry.acquire).toHaveBeenCalledOnce();
    unsubscribeFirst();
    expect(registry.release).not.toHaveBeenCalled();

    unsubscribeSecond();
    expect(resource.getSnapshot()).toBeNull();
    expect(registry.release).toHaveBeenCalledWith('project-a');
  });

  it('reacquires after the last subscriber releases the resource', () => {
    const engine = { projectId: 'project-a' } as CanvasEngine;
    registry.acquire.mockReturnValue(engine);
    const resource = createCanvasEngineResource('project-a', {} as never);

    resource.subscribe(() => undefined)();
    resource.subscribe(() => undefined)();

    expect(registry.acquire).toHaveBeenCalledTimes(2);
    expect(registry.release).toHaveBeenCalledTimes(2);
  });
});
