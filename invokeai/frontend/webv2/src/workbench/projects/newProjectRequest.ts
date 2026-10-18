import type { GalleryImage } from '@features/gallery';
import type { GenerateSettings } from '@features/generation/contracts';
import type { LaunchpadIntentId } from '@workbench/launchpad/intents';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';

/**
 * What the Launchpad's New-project dialog chose beyond the URL intent. The dialog stages it just before navigating to
 * `/app?new=true&intent=…`; the session controller takes it once, after applying the intent to the fresh draft.
 * Memory-only on purpose: a reload in between opens the plain intent draft, and a stale choice never resurfaces on a
 * later new-project link.
 */
export interface NewProjectRequest {
  intent: LaunchpadIntentId;
  /** Trimmed and nonempty; null keeps the default draft name. */
  name: string | null;
  /** Generate settings to commit after the intent is applied; null keeps the draft's defaults. */
  generateSettings: Partial<GenerateSettings> | null;
  /** A library workflow to load over the draft's blank workflow; null keeps the blank one. */
  libraryWorkflow: { workflowId: string; name: string } | null;
  /** An image a Canvas start opens on, as its first raster layer with the canvas sized to it; null starts blank. */
  startImage: GalleryImage | null;
}

let pending: NewProjectRequest | null = null;

registerAccountOwnedResource({
  clear: () => {
    pending = null;
  },
  name: 'new-project-request',
});

export const requestNewProject = (request: NewProjectRequest): void => {
  pending = request;
};

/** Returns and clears the pending request; null when none. */
export const takeNewProjectRequest = (): NewProjectRequest | null => {
  const request = pending;

  pending = null;

  return request;
};
