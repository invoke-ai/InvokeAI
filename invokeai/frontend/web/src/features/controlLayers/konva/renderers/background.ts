import { TRANSPARENCY_CHECKER_PATTERN } from 'features/controlLayers/konva/constants';
import { BACKGROUND_LAYER_ID, BACKGROUND_RECT_ID } from 'features/controlLayers/konva/naming';
import Konva from 'konva';
import { assert } from 'tsafe';

/**
 * The stage background is a semi-transparent checkerboard pattern. We use konva's `fillPatternImage` to apply the
 * a data URL of the pattern image to the background rect. Some scaling and positioning is required to ensure the
 * everything lines up correctly.
 */

/**
 * Creates the background layer for the stage.
 * @param stage The konva stage
 */
const createBackgroundLayer = (stage: Konva.Stage): Konva.Layer => {
  const layer = new Konva.Layer({
    id: BACKGROUND_LAYER_ID,
  });
  const background = new Konva.Rect({
    id: BACKGROUND_RECT_ID,
    x: stage.x(),
    y: 0,
    width: stage.width() / stage.scaleX(),
    height: stage.height() / stage.scaleY(),
    listening: false,
    opacity: 0.2,
  });
  layer.add(background);
  stage.add(layer);
  const image = new Image();
  image.onload = () => {
    background.fillPatternImage(image);
  };
  image.src = TRANSPARENCY_CHECKER_PATTERN;
  return layer;
};

/**
 * Renders the background layer for the stage.
 * @param stage The konva stage
 * @param width The unscaled width of the canvas
 * @param height The unscaled height of the canvas
 */
export const renderBackground = (stage: Konva.Stage, width: number, height: number): void => {
  const layer = stage.findOne<Konva.Layer>(`#${BACKGROUND_LAYER_ID}`) ?? createBackgroundLayer(stage);

  const background = layer.findOne<Konva.Rect>(`#${BACKGROUND_RECT_ID}`);
  assert(background, 'Background rect not found');
  // ensure background rect is in the top-left of the canvas
  background.absolutePosition({ x: 0, y: 0 });

  // set the dimensions of the background rect to match the canvas - not the stage!!!
  background.size({
    width: width / stage.scaleX(),
    height: height / stage.scaleY(),
  });

  // Calculate the amount the stage is moved - including the effect of scaling
  const stagePos = {
    x: -stage.x() / stage.scaleX(),
    y: -stage.y() / stage.scaleY(),
  };

  // Apply that movement to the fill pattern
  background.fillPatternOffset(stagePos);
};
