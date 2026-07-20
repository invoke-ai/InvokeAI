import type { ProviderComponent } from '@platform/react/composeProviders';

import { composeProviders } from '@platform/react/composeProviders';

import { GalleryUiAdapterProvider } from './GalleryUiAdapter';
import { GenerationUiAdapterProvider } from './GenerationUiAdapter';
import { ModelsUiAdapterProvider } from './ModelsUiAdapter';
import { QueueUiAdapterProvider } from './QueueUiAdapter';
import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';
import { WorkflowUiAdapterProvider } from './WorkflowUiAdapter';

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
