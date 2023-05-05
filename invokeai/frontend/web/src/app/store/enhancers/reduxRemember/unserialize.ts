import { canvasPersistDenylist } from 'features/canvas/store/canvasPersistDenylist';
import { initialCanvasState } from 'features/canvas/store/canvasSlice';
import { galleryPersistDenylist } from 'features/gallery/store/galleryPersistDenylist';
import { initialGalleryState } from 'features/gallery/store/gallerySlice';
import { resultsPersistDenylist } from 'features/gallery/store/resultsPersistDenylist';
import { initialResultsState } from 'features/gallery/store/resultsSlice';
import { uploadsPersistDenylist } from 'features/gallery/store/uploadsPersistDenylist';
import { initialUploadsState } from 'features/gallery/store/uploadsSlice';
import { lightboxPersistDenylist } from 'features/lightbox/store/lightboxPersistDenylist';
import { initialLightboxState } from 'features/lightbox/store/lightboxSlice';
import { nodesPersistDenylist } from 'features/nodes/store/nodesPersistDenylist';
import { initialNodesState } from 'features/nodes/store/nodesSlice';
import { generationPersistDenylist } from 'features/parameters/store/generationPersistDenylist';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import { postprocessingPersistDenylist } from 'features/parameters/store/postprocessingPersistDenylist';
import { initialPostprocessingState } from 'features/parameters/store/postprocessingSlice';
import { initialConfigState } from 'features/system/store/configSlice';
import { initialModelsState } from 'features/system/store/modelSlice';
import { modelsPersistDenylist } from 'features/system/store/modelsPersistDenylist';
import { systemPersistDenylist } from 'features/system/store/systemPersistDenylist';
import { initialSystemState } from 'features/system/store/systemSlice';
import { initialHotkeysState } from 'features/ui/store/hotkeysSlice';
import { uiPersistDenylist } from 'features/ui/store/uiPersistDenylist';
import { initialUIState } from 'features/ui/store/uiSlice';
import { defaultsDeep, merge, omit } from 'lodash-es';
import { UnserializeFunction } from 'redux-remember';

const initialStates: {
  [key: string]: any;
} = {
  canvas: initialCanvasState,
  gallery: initialGalleryState,
  generation: initialGenerationState,
  lightbox: initialLightboxState,
  models: initialModelsState,
  nodes: initialNodesState,
  postprocessing: initialPostprocessingState,
  results: initialResultsState,
  system: initialSystemState,
  config: initialConfigState,
  ui: initialUIState,
  uploads: initialUploadsState,
  hotkeys: initialHotkeysState,
};

export const unserialize: UnserializeFunction = (data, key) => {
  const result = defaultsDeep(JSON.parse(data), initialStates[key]);
  return result;
};
