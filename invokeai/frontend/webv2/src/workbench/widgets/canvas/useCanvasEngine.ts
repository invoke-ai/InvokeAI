/**
 * Resolves the shared {@link CanvasEngine} for the active project.
 *
 * The engine is a per-project resource shared with the layers widget via the
 * ref-counted {@link getOrCreateEngine | engine registry}. Acquisition and
 * release are a genuine component-lifecycle concern keyed on the project id —
 * the one place a `useEffect` is unavoidable here (there is no ref-callback for
 * "this component"). The registry's grace-period disposal absorbs the
 * mount/unmount churn (StrictMode double-mount, quick re-layout) so the same
 * warm engine is handed back rather than rebuilt.
 *
 * Returns `null` for the first render (before the acquiring effect runs); the
 * surface mounts once the engine is available, and its ref callback attaches
 * then.
 */

import type { ImageResolver } from '@workbench/canvas-engine/render/rasterizers';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';

import { getOrCreateEngine, releaseEngine } from '@workbench/canvas-operations/engineRegistry';
import { getImageFullUrl } from '@workbench/gallery/api';
import { useActiveProjectId, useWorkbenchStore } from '@workbench/WorkbenchContext';
import { useEffect, useState } from 'react';

/** Fetches a persisted image asset to a `Blob` for the engine's rasterizers. */
const createImageResolver = (): ImageResolver => async (imageName, signal) => {
  const response = await fetch(getImageFullUrl(imageName), { signal });
  if (!response.ok) {
    throw new Error(`Failed to load canvas image "${imageName}" (${response.status})`);
  }
  return response.blob();
};

/** Acquires (and releases) the active project's canvas engine. */
export const useCanvasEngine = (): CanvasEngine | null => {
  const store = useWorkbenchStore();
  const projectId = useActiveProjectId();
  const [engine, setEngine] = useState<CanvasEngine | null>(null);

  useEffect(() => {
    const acquired = getOrCreateEngine(projectId, { imageResolver: createImageResolver(), store });
    // Surfacing an effect-acquired external resource is the one legitimate
    // synchronous setState-in-effect: acquire/release must live in the effect
    // (balanced ref-count, StrictMode-safe via the registry's grace period),
    // and the surface can only attach once the handle reaches render.
    // eslint-disable-next-line react/react-compiler
    setEngine(acquired);
    return () => releaseEngine(projectId);
  }, [projectId, store]);

  return engine;
};
