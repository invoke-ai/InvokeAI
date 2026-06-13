import type { UserFont } from 'services/api/endpoints/utilities';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  awaitUserFontReady,
  buildCustomTextFontStacks,
  clearUserFontRegistryForTests,
  getUserFontFaceKey,
  isUserFontReady,
  syncUserFontFaces,
} from './textUserFonts';

describe('textUserFonts', () => {
  const face = {
    path: 'fonts/MyFont-Regular.ttf',
    url: '/api/v1/utilities/fonts/fonts/MyFont-Regular.ttf',
    weight: 400,
    style: 'normal' as const,
  };

  const font: UserFont = {
    id: 'user:fonts/MyFont-Regular.ttf',
    family: 'My Font',
    label: 'My Font',
    path: 'fonts/MyFont-Regular.ttf',
    url: '/api/v1/utilities/fonts/fonts/MyFont-Regular.ttf',
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
    const fontFaceCtor = vi.fn().mockImplementation(() => ({
      load: () => Promise.resolve(loadedFace),
    }));

    await syncUserFontFaces({
      fonts: [font],
      token: 'test-token',
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

    expect(fetchFn).toHaveBeenCalledWith(face.url, {
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
    const fontFaceCtor = vi.fn().mockImplementation(() => ({
      load: () => Promise.resolve(loadedFace),
    }));

    const syncPromise = syncUserFontFaces({
      fonts: [font],
      token: null,
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
    await readyPromise;

    expect(isUserFontReady(font.id)).toBe(true);
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
    const fontFaceCtor = vi.fn().mockImplementation(() => ({
      load: () => Promise.resolve(loadedFace),
    }));

    await syncUserFontFaces({
      fonts: [font],
      token: null,
      loadedFontFaces,
      fontFaceSet: {
        add: () => undefined,
        delete: () => true,
      },
      fontFaceCtor,
      fetchFn,
    });

    expect(isUserFontReady(font.id)).toBe(false);

    await syncUserFontFaces({
      fonts: [font],
      token: null,
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
