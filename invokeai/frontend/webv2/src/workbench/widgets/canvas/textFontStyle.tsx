import type { CanvasFontCapability, CanvasLayerSourceContract } from '@workbench/canvas-engine/api';

import { useMountEffect } from '@platform/react/useMountEffect';
import { useCallback, useSyncExternalStore } from 'react';

export type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;

export const textFontVariationSettings = (source: Pick<TextSource, 'fontVariations'>): string =>
  Object.entries(source.fontVariations ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([tag, value]) => `"${tag}" ${value}`)
    .join(', ');

export const textFontKey = (source: TextSource): string =>
  JSON.stringify([
    source.fontRef?.id ?? null,
    source.fontRef?.contentHash ?? null,
    source.fontFamily,
    source.fontStyle ?? 'normal',
    source.fontWeight,
    Object.entries(source.fontVariations ?? {})
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([tag, value]) => [tag, value]),
  ]);

/** Re-renders when the account-scoped runtime registers or evicts a custom face. */
export const useResolvedTextFontFamily = (fonts: CanvasFontCapability | undefined, source: TextSource): string => {
  const subscribe = useCallback((onChange: () => void) => fonts?.subscribe(onChange) ?? (() => {}), [fonts]);
  const getSnapshot = useCallback(() => fonts?.resolveFamily(source) ?? source.fontFamily, [fonts, source]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/** Keeps `source`'s face loading until it is replaced (key by {@link textFontKey}) or the owner unmounts. */
export const TextFontReadiness = ({
  fonts,
  source,
}: {
  fonts: CanvasFontCapability | undefined;
  source: TextSource;
}) => {
  useMountEffect(() => {
    if (!fonts || typeof fonts.ensurePreview !== 'function') {
      return;
    }
    const controller = new AbortController();
    void fonts.ensurePreview(source, controller.signal).catch(() => undefined);
    return () => controller.abort();
  });
  return null;
};
