import type { ProviderComponent } from '@platform/react/composeProviders';

import { composeProviders } from '@platform/react/composeProviders';

import { GalleryUiAdapterProvider } from './GalleryUiAdapter';
import { GenerationUiAdapterProvider } from './GenerationUiAdapter';
import { ModelsUiAdapterProvider } from './ModelsUiAdapter';
import { QueueUiAdapterProvider } from './QueueUiAdapter';
import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';
import { WorkflowUiAdapterProvider } from './WorkflowUiAdapter';

/**
 * The enumerable index of production port bindings: this file answers "what
 * does App bind in production?".
 *
 * Provider-shaped bindings are listed in `workbenchUiPortProviders` below.
 * The remaining production bindings are not providers and mount elsewhere:
 * - `QueueRuntimeAdapter` — mounted in `WorkbenchApp` (feeds Queue's runtime
 *   its Nodes execution store and Models load-activity sink).
 * - `SocketHubRuntime` — mounted in `router.tsx`'s AuthenticatedLayout, above
 *   the editor route.
 * - `GalleryImageActionsBridge` — lazy-loaded by `GalleryUiAdapter`.
 * - `configureHttpAuth(identityTransportAuthAdapter)` — called in `main.tsx`
 *   before React mounts.
 */
export const workbenchUiPortProviders: ReadonlyArray<ProviderComponent> = [
  ModelsUiAdapterProvider,
  QueueUiAdapterProvider,
  GalleryUiAdapterProvider,
  GenerationUiAdapterProvider,
  UpscaleUiAdapterProvider,
  WorkflowUiAdapterProvider,
];

/** All six feature UI ports, composed once at module scope (identity must be render-stable). */
export const WorkbenchUiPorts = composeProviders(workbenchUiPortProviders);
