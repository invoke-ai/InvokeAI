import { NO_LAYERS_MESSAGE_LAYER_ID } from 'features/controlLayers/konva/naming';
import { t } from 'i18next';
import Konva from 'konva';

/**
 * Logic for creating and rendering a fallback message when there are no layers to render.
 */

/**
 * Creates the "no layers" fallback layer
 * @param stage The konva stage
 */
const createNoLayersMessageLayer = (stage: Konva.Stage): Konva.Layer => {
  const noLayersMessageLayer = new Konva.Layer({
    id: NO_LAYERS_MESSAGE_LAYER_ID,
    opacity: 0.7,
    listening: false,
  });
  const text = new Konva.Text({
    x: 0,
    y: 0,
    align: 'center',
    verticalAlign: 'middle',
    text: t('controlLayers.noLayersAdded', 'No Layers Added'),
    fontFamily: '"Inter Variable", sans-serif',
    fontStyle: '600',
    fill: 'white',
  });
  noLayersMessageLayer.add(text);
  stage.add(noLayersMessageLayer);
  return noLayersMessageLayer;
};

/**
 * Renders the "no layers" message when there are no layers to render
 * @param stage The konva stage
 * @param layerCount The current number of layers
 * @param width The target width of the text
 * @param height The target height of the text
 */
export const renderNoLayersMessage = (stage: Konva.Stage, layerCount: number, width: number, height: number): void => {
  const noLayersMessageLayer =
    stage.findOne<Konva.Layer>(`#${NO_LAYERS_MESSAGE_LAYER_ID}`) ?? createNoLayersMessageLayer(stage);
  if (layerCount === 0) {
    noLayersMessageLayer.findOne<Konva.Text>('Text')?.setAttrs({
      width,
      height,
      fontSize: 32 / stage.scaleX(),
    });
  } else {
    noLayersMessageLayer?.destroy();
  }
};
