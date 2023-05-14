import { initialCanvasState } from 'features/canvas/store/canvasSlice';
import { initialGalleryState } from 'features/gallery/store/gallerySlice';
import { initialResultsState } from 'features/gallery/store/resultsSlice';
import { initialUploadsState } from 'features/gallery/store/uploadsSlice';
import { initialLightboxState } from 'features/lightbox/store/lightboxSlice';
import { initialNodesState } from 'features/nodes/store/nodesSlice';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import { initialPostprocessingState } from 'features/parameters/store/postprocessingSlice';
import { initialConfigState } from 'features/system/store/configSlice';
import { initialModelsState } from 'features/system/store/modelSlice';
import { initialSystemState } from 'features/system/store/systemSlice';
import { initialHotkeysState } from 'features/ui/store/hotkeysSlice';
import { initialUIState } from 'features/ui/store/uiSlice';
import { defaultsDeep } from 'lodash-es';
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
