import type { TextFontId } from 'features/controlLayers/text/textConstants';
import type { UserFont, UserFontFace } from 'services/api/endpoints/utilities';

type CustomTextFontStack = { id: TextFontId; label: string; stack: string };

type FetchResponseLike = {
  ok: boolean;
  arrayBuffer: () => Promise<ArrayBuffer>;
};

type FetchLike = (input: string, init?: RequestInit) => Promise<FetchResponseLike>;

type FontFaceLike<TLoadedFontFace> = {
  load: () => Promise<TLoadedFontFace>;
};

type FontFaceConstructorLike<TLoadedFontFace> = new (
  family: string,
  source: string | ArrayBuffer,
  descriptors?: { weight?: string; style?: string }
) => FontFaceLike<TLoadedFontFace>;

type FontFaceSetLike<TLoadedFontFace> = {
  add: (fontFace: TLoadedFontFace) => unknown;
  delete: (fontFace: TLoadedFontFace) => boolean;
};

export const buildCustomTextFontStacks = (fonts: Array<UserFont>): Array<CustomTextFontStack> => {
  return fonts.map((font) => ({
    id: font.id,
    label: font.label,
    stack: `"${font.family}",sans-serif`,
  }));
};

export const getUserFontFaceKey = (
  font: Pick<UserFont, 'family'>,
  face: Pick<UserFontFace, 'weight' | 'style' | 'url'>
) => `${font.family}|${face.weight}|${face.style}|${face.url}`;

type SyncUserFontFacesArgs<TLoadedFontFace> = {
  fonts: Array<UserFont>;
  token: string | null;
  loadedFontFaces: Map<string, TLoadedFontFace>;
  fontFaceSet: FontFaceSetLike<TLoadedFontFace>;
  fontFaceCtor: FontFaceConstructorLike<TLoadedFontFace>;
  fetchFn: FetchLike;
};

export async function syncUserFontFaces<TLoadedFontFace>({
  fonts,
  token,
  loadedFontFaces,
  fontFaceSet,
  fontFaceCtor,
  fetchFn,
}: SyncUserFontFacesArgs<TLoadedFontFace>): Promise<void> {
  const activeFaceKeys = new Set(fonts.flatMap((font) => font.faces.map((face) => getUserFontFaceKey(font, face))));

  for (const [faceKey, fontFace] of loadedFontFaces.entries()) {
    if (activeFaceKeys.has(faceKey)) {
      continue;
    }
    fontFaceSet.delete(fontFace);
    loadedFontFaces.delete(faceKey);
  }

  await Promise.all(
    fonts.flatMap((font) =>
      font.faces.map(async (face) => {
        const faceKey = getUserFontFaceKey(font, face);
        if (loadedFontFaces.has(faceKey)) {
          return;
        }

        try {
          const response = await fetchFn(face.url, {
            headers: token ? { Authorization: `Bearer ${token}` } : undefined,
          });
          if (!response.ok) {
            return;
          }

          const fontBuffer = await response.arrayBuffer();
          const fontFace = new fontFaceCtor(font.family, fontBuffer, {
            weight: String(face.weight),
            style: face.style,
          });
          const loadedFontFace = await fontFace.load();
          fontFaceSet.add(loadedFontFace);
          loadedFontFaces.set(faceKey, loadedFontFace);
        } catch {
          // Ignore failures and let browser fallback fonts render.
        }
      })
    )
  );
}
