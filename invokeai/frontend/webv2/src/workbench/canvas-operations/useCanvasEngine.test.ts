import type { CanvasEngine } from '@workbench/canvas-engine/api';

import { CanvasImageMissingError } from '@workbench/canvas-engine/api';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const registry = vi.hoisted(() => ({
  acquire: vi.fn(),
  release: vi.fn(),
}));

vi.mock('./engineRegistry', () => ({
  getOrCreateEngine: registry.acquire,
  releaseEngine: registry.release,
}));

import { createCanvasEngineResource, createImageResolver } from './useCanvasEngine';

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

describe('Canvas image resolver', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const resolveWith = (respond: () => Promise<Response>) => {
    vi.stubGlobal('fetch', vi.fn(respond));
    return createImageResolver()('cat.png');
  };

  it('returns the image body', async () => {
    const blob = await resolveWith(() => Promise.resolve(new Response('pixels', { status: 200 })));
    await expect(blob.text()).resolves.toBe('pixels');
  });

  it.each([403, 404, 410])('says a %i image is missing or not readable', async (status) => {
    const rejection = resolveWith(() => Promise.resolve(new Response(null, { status })));
    await expect(rejection).rejects.toBeInstanceOf(CanvasImageMissingError);
    await expect(rejection).rejects.toMatchObject({ imageName: 'cat.png' });
  });

  it.each([401, 429, 500, 503])('fails a %i as a plain error, which may pass', async (status) => {
    const rejection = resolveWith(() => Promise.resolve(new Response(null, { status })));
    await expect(rejection).rejects.toThrow(`(${status})`);
    await expect(rejection).rejects.not.toBeInstanceOf(CanvasImageMissingError);
  });

  it('passes a network error on as it is', async () => {
    const offline = new TypeError('Failed to fetch');
    await expect(resolveWith(() => Promise.reject(offline))).rejects.toBe(offline);
  });
});
