import type { RootState } from 'app/store/store';
import type { ImageDTO, NonNullableGraph } from 'services/api/types';

import { buildCanvasImageToImageGraph } from './buildCanvasImageToImageGraph';
import { buildCanvasInpaintGraph } from './buildCanvasInpaintGraph';
import { buildCanvasOutpaintGraph } from './buildCanvasOutpaintGraph';
import { buildCanvasSDXLImageToImageGraph } from './buildCanvasSDXLImageToImageGraph';
import { buildCanvasSDXLInpaintGraph } from './buildCanvasSDXLInpaintGraph';
import { buildCanvasSDXLOutpaintGraph } from './buildCanvasSDXLOutpaintGraph';
import { buildCanvasSDXLTextToImageGraph } from './buildCanvasSDXLTextToImageGraph';
import { buildCanvasTextToImageGraph } from './buildCanvasTextToImageGraph';

export const buildCanvasGraph = (
  state: RootState,
  generationMode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint',
  canvasInitImage: ImageDTO | undefined,
  canvasMaskImage: ImageDTO | undefined
) => {
  let graph: NonNullableGraph;

  if (generationMode === 'txt2img') {
    if (state.generation.model && state.generation.model.base_model === 'sdxl') {
      graph = buildCanvasSDXLTextToImageGraph(state);
    } else {
      graph = buildCanvasTextToImageGraph(state);
    }
  } else if (generationMode === 'img2img') {
    if (!canvasInitImage) {
      throw new Error('Missing canvas init image');
    }
    if (state.generation.model && state.generation.model.base_model === 'sdxl') {
      graph = buildCanvasSDXLImageToImageGraph(state, canvasInitImage);
    } else {
      graph = buildCanvasImageToImageGraph(state, canvasInitImage);
    }
  } else if (generationMode === 'inpaint') {
    if (!canvasInitImage || !canvasMaskImage) {
      throw new Error('Missing canvas init and mask images');
    }
    if (state.generation.model && state.generation.model.base_model === 'sdxl') {
      graph = buildCanvasSDXLInpaintGraph(state, canvasInitImage, canvasMaskImage);
    } else {
      graph = buildCanvasInpaintGraph(state, canvasInitImage, canvasMaskImage);
    }
  } else {
    if (!canvasInitImage) {
      throw new Error('Missing canvas init image');
    }
    if (state.generation.model && state.generation.model.base_model === 'sdxl') {
      graph = buildCanvasSDXLOutpaintGraph(state, canvasInitImage, canvasMaskImage);
    } else {
      graph = buildCanvasOutpaintGraph(state, canvasInitImage, canvasMaskImage);
    }
  }

  return graph;
};
