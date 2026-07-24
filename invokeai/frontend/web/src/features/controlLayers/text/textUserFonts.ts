import { isCustomTextFontId, type TextFontId } from 'features/controlLayers/text/textConstants';
import { atom } from 'nanostores';
import type { UserFont, UserFontFace } from 'services/api/endpoints/utilities';

type CustomTextFontStack = { id: TextFontId; label: string; stack: string };
type UserFontReadyState = 'pending' | 'ready' | 'error';

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

export const $userFontReadyStates = atom<Record<TextFontId, UserFontReadyState>>({});
const userFontReadyPromises = new Map<TextFontId, Promise<void>>();
const userFontReadyResolvers = new Map<TextFontId, () => void>();
const USER_FONT_READY_TIMEOUT_MS = 2000;

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

const areAllFontFacesLoaded = <TLoadedFontFace>(
  font: UserFont,
  loadedFontFaces: Map<string, TLoadedFontFace>
): boolean => {
  return font.faces.every((face) => loadedFontFaces.has(getUserFontFaceKey(font, face)));
};

const setUserFontReadyState = (fontId: TextFontId, state: UserFontReadyState): void => {
  const currentStates = $userFontReadyStates.get();
  if (currentStates[fontId] === state) {
    return;
  }
  $userFontReadyStates.set({
    ...currentStates,
    [fontId]: state,
  });
};

const deleteUserFontReadyState = (fontId: TextFontId): void => {
  const currentStates = $userFontReadyStates.get();
  if (!(fontId in currentStates)) {
    return;
  }
  const nextStates = { ...currentStates };
  delete nextStates[fontId];
  $userFontReadyStates.set(nextStates);
};

const ensureUserFontReadyPromise = (fontId: TextFontId): Promise<void> => {
  const existingPromise = userFontReadyPromises.get(fontId);
  if (existingPromise) {
    return existingPromise;
  }

  let resolvePromise: (() => void) | undefined;
  const promise = new Promise<void>((resolve) => {
    resolvePromise = resolve;
  });
  userFontReadyPromises.set(fontId, promise);
  userFontReadyResolvers.set(fontId, resolvePromise ?? (() => undefined));
  return promise;
};

const settleUserFontReadyPromise = (fontId: TextFontId): void => {
  userFontReadyResolvers.get(fontId)?.();
  userFontReadyResolvers.delete(fontId);
  userFontReadyPromises.delete(fontId);
};

export const primeUserFontReadiness = <TLoadedFontFace>(
  fonts: Array<UserFont>,
  loadedFontFaces: Map<string, TLoadedFontFace>
): void => {
  const activeFontIds = new Set(fonts.map((font) => font.id));

  for (const fontId of Object.keys($userFontReadyStates.get()) as TextFontId[]) {
    if (isCustomTextFontId(fontId) && !activeFontIds.has(fontId)) {
      deleteUserFontReadyState(fontId);
      settleUserFontReadyPromise(fontId);
    }
  }

  for (const font of fonts) {
    if (areAllFontFacesLoaded(font, loadedFontFaces)) {
      setUserFontReadyState(font.id, 'ready');
      settleUserFontReadyPromise(font.id);
      continue;
    }

    setUserFontReadyState(font.id, 'pending');
    ensureUserFontReadyPromise(font.id);
  }
};

export const isUserFontReady = (fontId: TextFontId): boolean => {
  return !isCustomTextFontId(fontId) || $userFontReadyStates.get()[fontId] === 'ready';
};

export const awaitUserFontReady = async (fontId: TextFontId): Promise<void> => {
  if (!isCustomTextFontId(fontId)) {
    return;
  }
  const state = $userFontReadyStates.get()[fontId];
  if (state === 'ready' || state === 'error') {
    return;
  }
  await Promise.race([
    ensureUserFontReadyPromise(fontId),
    new Promise<void>((resolve) => {
      setTimeout(resolve, USER_FONT_READY_TIMEOUT_MS);
    }),
  ]);
};

export const clearUserFontRegistryForTests = (): void => {
  $userFontReadyStates.set({});
  userFontReadyPromises.clear();
  userFontReadyResolvers.clear();
};

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
    fonts.map(async (font) => {
      if (areAllFontFacesLoaded(font, loadedFontFaces)) {
        setUserFontReadyState(font.id, 'ready');
        settleUserFontReadyPromise(font.id);
        return;
      }

      setUserFontReadyState(font.id, 'pending');
      ensureUserFontReadyPromise(font.id);
      let hadFailure = false;

      await Promise.all(
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
              hadFailure = true;
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
            hadFailure = true;
          }
        })
      );

      if (areAllFontFacesLoaded(font, loadedFontFaces)) {
        setUserFontReadyState(font.id, 'ready');
      } else if (hadFailure) {
        setUserFontReadyState(font.id, 'error');
      }

      if ($userFontReadyStates.get()[font.id] !== 'pending') {
        settleUserFontReadyPromise(font.id);
      }
    })
  );
}
