import type { UserFont } from 'services/api/endpoints/utilities';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  awaitUserFontReady,
  buildCustomTextFontStacks,
  clearUserFontRegistryForTests,
  getUserFontFaceKey,
  isUserFontReady,
  loadedUserFontFaces,
  syncUserFontFaces,
} from './textUserFonts';

describe('textUserFonts', () => {
  const face = {
    path: 'fonts/MyFont-Regular.ttf',
    url: 'api/v1/utilities/fonts/fonts/MyFont-Regular.ttf',
    weight: 400,
    style: 'normal' as const,
  };

  const font: UserFont = {
    id: 'user:fonts/MyFont-Regular.ttf',
    family: 'My Font',
    label: 'My Font',
    path: 'fonts/MyFont-Regular.ttf',
    url: 'api/v1/utilities/fonts/fonts/MyFont-Regular.ttf',
    faces: [face],
  };

  afterEach(() => {
    clearUserFontRegistryForTests();
  });

  it('builds custom font stacks from user fonts', () => {
    expect(buildCustomTextFontStacks([font])).toEqual([
      {
        id: font.id,
        label: font.label,
        stack: '"My Font",sans-serif',
      },
    ]);
  });

  it('loads authenticated font faces and prunes stale entries', async () => {
    const staleFace = { family: 'Stale Font' };
    const loadedFontFaces = new Map<string, object>([['stale|400|normal|/stale.ttf', staleFace]]);
    const addedFaces: object[] = [];
    const deletedFaces: object[] = [];
    const loadedFace = { family: 'My Font' };
    const fetchFn = vi.fn(() =>
      Promise.resolve({
        ok: true,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
      })
    );
    const fontFaceCtor = vi.fn(function () {
      return {
        load: () => Promise.resolve(loadedFace),
      };
    });

    await syncUserFontFaces({
      fonts: [font],
      token: 'test-token',
      baseUrl: 'https://invoke.example.com/subpath',
      loadedFontFaces,
      fontFaceSet: {
        add: (face) => addedFaces.push(face),
        delete: (face) => {
          deletedFaces.push(face);
          return true;
        },
      },
      fontFaceCtor,
      fetchFn,
    });

    const faceKey = getUserFontFaceKey(font, face);

    expect(fetchFn).toHaveBeenCalledWith(`https://invoke.example.com/subpath/${face.url}`, {
      headers: { Authorization: 'Bearer test-token' },
    });
    expect(fontFaceCtor).toHaveBeenCalledWith('My Font', expect.any(ArrayBuffer), {
      weight: '400',
      style: 'normal',
    });
    expect(addedFaces).toEqual([loadedFace]);
    expect(deletedFaces).toEqual([staleFace]);
    expect(loadedFontFaces.get(faceKey)).toBe(loadedFace);
    expect(loadedFontFaces.has('stale|400|normal|/stale.ttf')).toBe(false);
  });

  it('reuses the module-level font face registry across sync cycles', async () => {
    const loadedFace = { family: 'My Font' } as FontFace;
    const addedFaces: FontFace[] = [];
    const fetchFn = vi.fn(() =>
      Promise.resolve({
        ok: true,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
      })
    );
    const fontFaceCtor = vi.fn(function () {
      return {
        load: () => Promise.resolve(loadedFace),
      };
    });
    const args = {
      fonts: [font],
      token: null,
      baseUrl: 'https://invoke.example.com',
      loadedFontFaces: loadedUserFontFaces,
      fontFaceSet: {
        add: (loadedFace: FontFace) => addedFaces.push(loadedFace),
        delete: () => true,
      },
      fontFaceCtor,
      fetchFn,
    };

    await syncUserFontFaces(args);
    await syncUserFontFaces(args);

    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(addedFaces).toEqual([loadedFace]);
  });

  it('tracks custom font readiness until all faces load', async () => {
    const loadedFontFaces = new Map<string, object>();
    const loadedFace = { family: 'My Font' };
    let resolveFetch: ((response: { ok: boolean; arrayBuffer: () => Promise<ArrayBuffer> }) => void) | undefined;
    const fetchFn = vi.fn(
      () =>
        new Promise<{ ok: boolean; arrayBuffer: () => Promise<ArrayBuffer> }>((resolve) => {
          resolveFetch = resolve;
        })
    );
    const fontFaceCtor = vi.fn(function () {
      return {
        load: () => Promise.resolve(loadedFace),
      };
    });

    const syncPromise = syncUserFontFaces({
      fonts: [font],
      token: null,
      baseUrl: 'https://invoke.example.com',
      loadedFontFaces,
      fontFaceSet: {
        add: () => undefined,
        delete: () => true,
      },
      fontFaceCtor,
      fetchFn,
    });

    expect(isUserFontReady(font.id)).toBe(false);

    const readyPromise = awaitUserFontReady(font.id);
    let isReadyResolved = false;
    void readyPromise.then(() => {
      isReadyResolved = true;
    });

    await Promise.resolve();
    expect(isReadyResolved).toBe(false);

    resolveFetch?.({
      ok: true,
      arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
    });

    await syncPromise;
    await expect(readyPromise).resolves.toBe('ready');

    expect(isUserFontReady(font.id)).toBe(true);
  });

  it('reports timeout while a custom font is still pending', async () => {
    vi.useFakeTimers();

    try {
      const loadedFontFaces = new Map<string, object>();
      const fetchFn = vi.fn(
        () =>
          new Promise<{ ok: boolean; arrayBuffer: () => Promise<ArrayBuffer> }>(() => {
            // Keep the font request pending so readiness must time out.
          })
      );
      const fontFaceCtor = vi.fn(function () {
        return {
          load: () => Promise.resolve({ family: 'My Font' }),
        };
      });

      void syncUserFontFaces({
        fonts: [font],
        token: null,
        baseUrl: 'https://invoke.example.com',
        loadedFontFaces,
        fontFaceSet: {
          add: () => undefined,
          delete: () => true,
        },
        fontFaceCtor,
        fetchFn,
      });

      await Promise.resolve();
      const readinessPromise = awaitUserFontReady(font.id);
      await vi.advanceTimersByTimeAsync(2000);

      await expect(readinessPromise).resolves.toBe('timeout');
      expect(isUserFontReady(font.id)).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
  it('allows a later sync to recover from an initial load failure', async () => {
    const loadedFontFaces = new Map<string, object>();
    const loadedFace = { family: 'My Font' };
    const fetchFn = vi
      .fn<() => Promise<{ ok: boolean; arrayBuffer: () => Promise<ArrayBuffer> }>>()
      .mockResolvedValueOnce({
        ok: false,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
      })
      .mockResolvedValueOnce({
        ok: true,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
      });
    const fontFaceCtor = vi.fn(function () {
      return {
        load: () => Promise.resolve(loadedFace),
      };
    });

    await syncUserFontFaces({
      fonts: [font],
      token: null,
      baseUrl: 'https://invoke.example.com',
      loadedFontFaces,
      fontFaceSet: {
        add: () => undefined,
        delete: () => true,
      },
      fontFaceCtor,
      fetchFn,
    });

    expect(isUserFontReady(font.id)).toBe(false);
    await expect(awaitUserFontReady(font.id)).resolves.toBe('error');

    await syncUserFontFaces({
      fonts: [font],
      token: null,
      baseUrl: 'https://invoke.example.com',
      loadedFontFaces,
      fontFaceSet: {
        add: () => undefined,
        delete: () => true,
      },
      fontFaceCtor,
      fetchFn,
    });

    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(isUserFontReady(font.id)).toBe(true);
  });
});
