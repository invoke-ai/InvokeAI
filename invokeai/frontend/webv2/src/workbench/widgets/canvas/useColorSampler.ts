/**
 * Binds a color picker's eyedropper to the canvas.
 *
 * The browser's `EyeDropper` API only reads the composited *screen*, which is
 * Chromium-only and picks up whatever chrome happens to sit over the canvas.
 * The engine already owns a better sampler — the color-picker tool reads the
 * composited document through the layer cache — so any picker rendered
 * alongside a canvas hands this to `ColorPicker`'s `onSampleColor` instead.
 *
 * The returned callback is stable, which the `react-perf/jsx-no-new-*-as-prop`
 * rules require of anything passed straight into JSX.
 */

import type { CanvasEngineToolCapability } from '@workbench/canvas-engine/api';

import { useCallback } from 'react';

/** The narrowest engine slice this needs, so callers holding a `Pick<>` still fit. */
export interface ColorSamplerEngine {
  readonly tools: Pick<CanvasEngineToolCapability, 'requestColorSample'>;
}

export const useColorSampler = (engine: ColorSamplerEngine): (() => Promise<string | null>) =>
  useCallback(() => engine.tools.requestColorSample(), [engine]);
