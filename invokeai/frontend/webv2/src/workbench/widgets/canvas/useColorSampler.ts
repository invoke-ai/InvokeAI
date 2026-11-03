/**
 * Prefer the engine's document sampler over Chromium's composited-screen eyedropper. Return a stable callback with
 * an engine, otherwise preserve ColorPicker's fallback.
 */

import type { CanvasEngineToolCapability } from '@workbench/canvas-engine/api';

import { useCallback } from 'react';

/** The narrowest engine slice this needs, so callers holding a `Pick<>` still fit. */
export interface ColorSamplerEngine {
  readonly tools: Pick<CanvasEngineToolCapability, 'requestColorSample'>;
}

export const useColorSampler = (engine: ColorSamplerEngine | null): (() => Promise<string | null>) | undefined => {
  const sample = useCallback(() => engine?.tools.requestColorSample() ?? Promise.resolve(null), [engine]);
  return engine ? sample : undefined;
};
