import type { UserFont } from 'services/api/endpoints/utilities';
import { describe, expect, it, vi } from 'vitest';

import { buildCustomTextFontStacks, getUserFontFaceKey, syncUserFontFaces } from './textUserFonts';

describe('textUserFonts', () => {
  const face = {
    path: 'Fonts/MyFont-Regular.ttf',
    url: '/api/v1/utilities/fonts/Fonts/MyFont-Regular.ttf',
    weight: 400,
    style: 'normal' as const,
  };

  const font: UserFont = {
    id: 'user:Fonts/MyFont-Regular.ttf',
    family: 'My Font',
    label: 'My Font',
    path: 'Fonts/MyFont-Regular.ttf',
    url: '/api/v1/utilities/fonts/Fonts/MyFont-Regular.ttf',
    faces: [face],
  };

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
});
