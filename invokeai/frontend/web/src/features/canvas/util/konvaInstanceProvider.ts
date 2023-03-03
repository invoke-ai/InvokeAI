import Konva from 'konva';

let canvasBaseLayer: Konva.Layer | null = null;
let canvasStage: Konva.Stage | null = null;

export const setCanvasBaseLayer = (layer: Konva.Layer) => {
  canvasBaseLayer = layer;
};

export const getCanvasBaseLayer = () => canvasBaseLayer;

export const setCanvasStage = (stage: Konva.Stage) => {
  canvasStage = stage;
};

export const getCanvasStage = () => canvasStage;
