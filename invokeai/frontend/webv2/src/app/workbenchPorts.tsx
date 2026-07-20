import type { ProviderComponent } from '@platform/react/composeProviders';

import { composeProviders } from '@platform/react/composeProviders';

import { GalleryUiAdapterProvider } from './GalleryUiAdapter';
import { GenerationUiAdapterProvider } from './GenerationUiAdapter';
import { ModelsUiAdapterProvider } from './ModelsUiAdapter';
import { bindProductionPortBinding } from './productionPortBindings';
import { QueueUiAdapterProvider } from './QueueUiAdapter';
import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';
import { WorkflowUiAdapterProvider } from './WorkflowUiAdapter';

export const workbenchUiPortProviders: ReadonlyArray<ProviderComponent> = [
  bindProductionPortBinding('models.ui', ModelsUiAdapterProvider),
  bindProductionPortBinding('queue.ui', QueueUiAdapterProvider),
  bindProductionPortBinding('gallery.ui', GalleryUiAdapterProvider),
  bindProductionPortBinding('generation.ui', GenerationUiAdapterProvider),
  bindProductionPortBinding('upscale.ui', UpscaleUiAdapterProvider),
  bindProductionPortBinding('workflow.ui', WorkflowUiAdapterProvider),
];

/** All six feature UI ports, composed once at module scope (identity must be render-stable). */
export const WorkbenchUiPorts = composeProviders(workbenchUiPortProviders);
