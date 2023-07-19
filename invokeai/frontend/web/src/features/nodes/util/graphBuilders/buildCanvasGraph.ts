import { log } from 'app/logging/useLogger';
import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { ImageDTO } from 'services/api/types';
import { buildCanvasImageToImageGraph } from './buildCanvasImageToImageGraph';
import { buildCanvasInpaintGraph } from './buildCanvasInpaintGraph';
import { buildCanvasTextToImageGraph } from './buildCanvasTextToImageGraph';

const moduleLog = log.child({ namespace: 'nodes' });

export const buildCanvasGraph = (
  state: RootState,
  generationMode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint',
  canvasInitImage: ImageDTO | undefined,
  canvasMaskImage: ImageDTO | undefined
) => {
  let graph: NonNullableGraph;

  if (generationMode === 'txt2img') {
    graph = buildCanvasTextToImageGraph(state);
  } else if (generationMode === 'img2img') {
    if (!canvasInitImage) {
      throw new Error('Missing canvas init image');
    }
    graph = buildCanvasImageToImageGraph(state, canvasInitImage);
  } else {
    if (!canvasInitImage || !canvasMaskImage) {
      throw new Error('Missing canvas init and mask images');
    }
    graph = buildCanvasInpaintGraph(state, canvasInitImage, canvasMaskImage);
  }

  return graph;
};
