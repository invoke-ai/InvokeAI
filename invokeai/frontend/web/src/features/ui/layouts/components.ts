import type { IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { CanvasLayersPanel } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { GenerateLaunchpadPanel } from 'features/controlLayers/components/SimpleSession/GenerateLaunchpadPanel';
import { BoardsPanel } from 'features/gallery/components/BoardsListPanelContent';
import { GalleryPanel } from 'features/gallery/components/Gallery';
import { GenerationProgressPanel } from 'features/gallery/components/ImageViewer/GenerationProgressPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { CanvasWorkspacePanel } from 'features/ui/layouts/canvas-tab-auto-layout';
import { GenerateLeftPanel } from 'features/ui/layouts/generate-tab-auto-layout';

export const components: IDockviewReactProps['components'] & IGridviewReactProps['components'] = {
  // Shared components
  Gallery: GalleryPanel,
  Boards: BoardsPanel,
  ImageViewer: ImageViewerPanel,
  GenerationProgress: GenerationProgressPanel,
  // Generate tab
  GenerateLaunchpad: GenerateLaunchpadPanel,
  GenerateLeft: GenerateLeftPanel,
  // Upscaling tab
  UpscalingLaunchpad: GenerateLaunchpadPanel,
  // Workflows tab
  WorkflowsLaunchpad: GenerateLaunchpadPanel,
  // Canvas tab
  CanvasLaunchpad: GenerateLaunchpadPanel,
  CanvasLayers: CanvasLayersPanel,
  CanvasWorkspace: CanvasWorkspacePanel,
  CanvasLeft: GenerateLeftPanel,
};
