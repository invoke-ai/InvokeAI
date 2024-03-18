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

export const buildCanvasGraph = async (
  state: RootState,
  generationMode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint',
  canvasInitImage: ImageDTO | undefined,
  canvasMaskImage: ImageDTO | undefined
): Promise<NonNullableGraph> => {
  let graph: NonNullableGraph;

  if (generationMode === 'txt2img') {
    if (state.generation.model && state.generation.model.base === 'sdxl') {
      graph = await buildCanvasSDXLTextToImageGraph(state);
    } else {
      graph = await buildCanvasTextToImageGraph(state);
    }
  } else if (generationMode === 'img2img') {
    if (!canvasInitImage) {
      throw new Error('Missing canvas init image');
    }
    if (state.generation.model && state.generation.model.base === 'sdxl') {
      graph = await buildCanvasSDXLImageToImageGraph(state, canvasInitImage);
    } else {
      graph = await buildCanvasImageToImageGraph(state, canvasInitImage);
    }
  } else if (generationMode === 'inpaint') {
    if (!canvasInitImage || !canvasMaskImage) {
      throw new Error('Missing canvas init and mask images');
    }
    if (state.generation.model && state.generation.model.base === 'sdxl') {
      graph = await buildCanvasSDXLInpaintGraph(state, canvasInitImage, canvasMaskImage);
    } else {
      graph = await buildCanvasInpaintGraph(state, canvasInitImage, canvasMaskImage);
    }
  } else {
    if (!canvasInitImage) {
      throw new Error('Missing canvas init image');
    }
    if (state.generation.model && state.generation.model.base === 'sdxl') {
      graph = await buildCanvasSDXLOutpaintGraph(state, canvasInitImage, canvasMaskImage);
    } else {
      graph = await buildCanvasOutpaintGraph(state, canvasInitImage, canvasMaskImage);
    }
  }

  return graph;
};
