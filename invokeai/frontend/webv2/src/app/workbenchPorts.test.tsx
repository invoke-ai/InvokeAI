import { LivePreviewFollowProvider } from '@workbench/widgets/preview/livePreviewFollow';
import { describe, expect, it } from 'vitest';

import { GalleryUiAdapterProvider } from './GalleryUiAdapter';
import { GenerationUiAdapterProvider } from './GenerationUiAdapter';
import { ModelsUiAdapterProvider } from './ModelsUiAdapter';
import { QueueUiAdapterProvider } from './QueueUiAdapter';
import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';
import { VideoUiAdapterProvider } from './VideoUiAdapter';
import { workbenchUiPortProviders } from './workbenchPorts';
import { WorkflowUiAdapterProvider } from './WorkflowUiAdapter';

describe('workbench UI port composition', () => {
  it('composes every feature adapter once in stable provider order', () => {
    expect(workbenchUiPortProviders).toEqual([
      LivePreviewFollowProvider,
      ModelsUiAdapterProvider,
      QueueUiAdapterProvider,
      GalleryUiAdapterProvider,
      GenerationUiAdapterProvider,
      UpscaleUiAdapterProvider,
      VideoUiAdapterProvider,
      WorkflowUiAdapterProvider,
    ]);
  });
});
