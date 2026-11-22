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

const resourceEngine = (hasVectors = false) => {
  let status = 'idle';
  const listeners = new Map<string, Set<() => void>>();
  const publish = (key: string) => listeners.get(key)?.forEach((listener) => listener());
  const setStatus = (value: string) => {
    status = value;
    publish('vectorToolsStatus');
  };
  const prepare = vi.fn(() => {
    setStatus('loading');
    return Promise.resolve(true);
  });
  const engine = {
    projectId: 'project-a',
    document: { getDocument: () => ({ stacks: { vector: hasVectors ? [{}] : [] } }) },
    interaction: {
      get: () => status,
      subscribe: (key: string, listener: () => void) => {
        const subscribers = listeners.get(key) ?? new Set<() => void>();
        listeners.set(key, subscribers);
        subscribers.add(listener);
        return () => subscribers.delete(listener);
      },
    },
    layers: { vectorEdit: { prepare } },
  } as unknown as CanvasEngine;
  return { engine, prepare, setStatus, listeners };
};

describe('Canvas engine React resource', () => {
  beforeEach(() => {
    registry.acquire.mockReset();
    registry.release.mockReset();
  });

  it('acquires only while subscribed and balances one lease across subscribers', () => {
    const { engine } = resourceEngine();
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
    const { engine } = resourceEngine();
    registry.acquire.mockReturnValue(engine);
    const resource = createCanvasEngineResource('project-a', {} as never);

    resource.subscribe(() => undefined)();
    resource.subscribe(() => undefined)();

    expect(registry.acquire).toHaveBeenCalledTimes(2);
    expect(registry.release).toHaveBeenCalledTimes(2);
  });
  it('holds a persisted vector document until its tools are ready and removes subscriptions on release', () => {
    const { engine, prepare, setStatus, listeners } = resourceEngine(true);
    registry.acquire.mockReturnValue(engine);
    const resource = createCanvasEngineResource('project-a', {} as never);
    const changed = vi.fn();
    const release = resource.subscribe(changed);
    expect(prepare).toHaveBeenCalledOnce();
    expect(resource.getSnapshot()).toBeNull();
    setStatus('ready');
    expect(resource.getSnapshot()).toBe(engine);
    expect(changed).toHaveBeenCalled();
    release();
    expect([...listeners.values()].every((set) => set.size === 0)).toBe(true);
    changed.mockClear();
    setStatus('loading');
    expect(changed).not.toHaveBeenCalled();
  });

  it('leaves raster documents usable without requesting vector support', () => {
    const { engine, prepare } = resourceEngine();
    registry.acquire.mockReturnValue(engine);
    const resource = createCanvasEngineResource('project-a', {} as never);
    const release = resource.subscribe(() => undefined);
    expect(resource.getSnapshot()).toBe(engine);
    expect(prepare).not.toHaveBeenCalled();
    release();
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
