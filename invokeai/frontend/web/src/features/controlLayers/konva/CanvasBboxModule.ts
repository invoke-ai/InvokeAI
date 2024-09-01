import { roundToMultiple, roundToMultipleMin } from 'common/util/roundDownToMultiple';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Coordinate, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { assert } from 'tsafe';

const ALL_ANCHORS: string[] = [
  'top-left',
  'top-center',
  'top-right',
  'middle-right',
  'middle-left',
  'bottom-left',
  'bottom-center',
  'bottom-right',
];
const CORNER_ANCHORS: string[] = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
const NO_ANCHORS: string[] = [];

/**
 * Renders the bounding box. The bounding box can be transformed by the user.
 */
export class CanvasBboxModule extends CanvasModuleBase {
  readonly type = 'bbox';

  id: string;
  path: string[];
  parent: CanvasManager;
  manager: CanvasManager;
  log: Logger;

  subscriptions: Set<() => void> = new Set();

  /**
   * The Konva objects that make up the bbox:
   * - A group to hold all the objects
   * - A transformer to allow the bbox to be transformed
   * - A transparent rect so the transformer has something to transform
   */
  konva: {
    group: Konva.Group;
    transformer: Konva.Transformer;
    proxyRect: Konva.Rect;
  };

  /**
   * Buffer to store the last aspect ratio of the bbox. When the users holds shift while transforming the bbox, this is
   * used to lock the aspect ratio.
   */
  $aspectRatioBuffer = atom(0);

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating bbox module');

    // Set the initial aspect ratio buffer per app state.
    const bbox = this.manager.stateApi.getBbox();
    this.$aspectRatioBuffer.set(bbox.rect.width / bbox.rect.height);

    this.konva = {
      group: new Konva.Group({ name: `${this.type}:group`, listening: true }),
      // We will use a Konva.Transformer for the generation bbox. Transformers need some shape to transform, so we will
      // create a transparent rect for this purpose.
      proxyRect: new Konva.Rect({
        name: `${this.type}:rect`,
        listening: false,
        strokeEnabled: false,
        draggable: true,
        x: bbox.rect.x,
        y: bbox.rect.y,
        width: bbox.rect.width,
        height: bbox.rect.height,
      }),
      transformer: new Konva.Transformer({
        name: `${this.type}:transformer`,
        borderDash: [5, 5],
        borderStroke: 'rgba(212,216,234,1)',
        borderEnabled: true,
        rotateEnabled: false,
        keepRatio: false,
        ignoreStroke: true,
        listening: false,
        flipEnabled: false,
        anchorFill: 'rgba(212,216,234,1)',
        anchorStroke: 'rgb(42,42,42)',
        anchorSize: 12,
        anchorCornerRadius: 3,
        shiftBehavior: 'none', // we will implement our own shift behavior
        centeredScaling: false,
        anchorStyleFunc: this.anchorStyleFunc,
        anchorDragBoundFunc: this.anchorDragBoundFunc,
      }),
    };

    this.konva.proxyRect.on('dragmove', this.onDragMove);
    this.konva.transformer.on('transform', this.onTransform);
    this.konva.transformer.on('transformend', this.onTransformEnd);

    // The transformer will always be transforming the proxy rect
    this.konva.transformer.nodes([this.konva.proxyRect]);
    this.konva.group.add(this.konva.proxyRect);
    this.konva.group.add(this.konva.transformer);

    // We will listen to the tool state to determine if the bbox should be visible or not.
    this.subscriptions.add(this.manager.stateApi.$tool.listen(this.render));
  }

  /**
   * Renders the bbox. The bbox is only visible when the tool is set to 'bbox'.
   */
  render = () => {
    this.log.trace('Rendering');

    const { x, y, width, height } = this.manager.stateApi.getBbox().rect;
    const tool = this.manager.stateApi.$tool.get();

    this.konva.group.visible(true);

    // We need to reach up to the preview layer to enable/disable listening so that the bbox can be interacted with.
    this.manager.konva.previewLayer.listening(tool === 'bbox');

    this.konva.proxyRect.setAttrs({
      x,
      y,
      width,
      height,
      scaleX: 1,
      scaleY: 1,
      listening: tool === 'bbox',
    });
    this.konva.transformer.setAttrs({
      listening: tool === 'bbox',
      enabledAnchors: tool === 'bbox' ? ALL_ANCHORS : NO_ANCHORS,
    });
  };

  destroy = () => {
    this.log.trace('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.konva.group.destroy();
  };

  /**
   * Handles the dragmove event on the bbox rect:
   * - Snaps the bbox position to the grid (determined by ctrl/meta key)
   * - Pushes the new bbox rect into app state
   */
  onDragMove = () => {
    const gridSize = this.manager.stateApi.$ctrlKey.get() || this.manager.stateApi.$metaKey.get() ? 8 : 64;
    const bbox = this.manager.stateApi.getBbox();
    const bboxRect: Rect = {
      ...bbox.rect,
      x: roundToMultiple(this.konva.proxyRect.x(), gridSize),
      y: roundToMultiple(this.konva.proxyRect.y(), gridSize),
    };
    this.konva.proxyRect.setAttrs(bboxRect);
    if (bbox.rect.x !== bboxRect.x || bbox.rect.y !== bboxRect.y) {
      this.manager.stateApi.setGenerationBbox(bboxRect);
    }
  };

  /**
   * Handles the transform event on the bbox transformer:
   * - Snaps the bbox dimensions to the grid (determined by ctrl/meta key)
   * - Centered scaling when alt is held
   * - Aspect ratio locking when shift is held
   * - Pushes the new bbox rect into app state
   * - Syncs the aspect ratio buffer
   */
  onTransform = () => {
    // In the transform callback, we calculate the bbox's new dims and pos and update the konva object.
    // Some special handling is needed depending on the anchor being dragged.
    const anchor = this.konva.transformer.getActiveAnchor();
    if (!anchor) {
      // Pretty sure we should always have an anchor here?
      return;
    }

    const alt = this.manager.stateApi.$altKey.get();
    const ctrl = this.manager.stateApi.$ctrlKey.get();
    const meta = this.manager.stateApi.$metaKey.get();
    const shift = this.manager.stateApi.$shiftKey.get();

    // Grid size depends on the modifier keys
    let gridSize = ctrl || meta ? 8 : 64;

    // Alt key indicates we are using centered scaling. We need to double the gride size used when calculating the
    // new dimensions so that each size scales in the correct increments and doesn't mis-place the bbox. For example, if
    // we snapped the width and height to 8px increments, the bbox would be mis-placed by 4px in the x and y axes.
    // Doubling the grid size ensures the bbox's coords remain aligned to the 8px/64px grid.
    if (this.manager.stateApi.$altKey.get()) {
      gridSize = gridSize * 2;
    }

    // The coords should be correct per the anchorDragBoundFunc.
    let x = this.konva.proxyRect.x();
    let y = this.konva.proxyRect.y();

    // Konva transforms by scaling the dims, not directly changing width and height. At this point, the width and height
    // *have not changed*, only the scale has changed. To get the final height, we need to scale the dims and then snap
    // them to the grid.
    let width = roundToMultipleMin(this.konva.proxyRect.width() * this.konva.proxyRect.scaleX(), gridSize);
    let height = roundToMultipleMin(this.konva.proxyRect.height() * this.konva.proxyRect.scaleY(), gridSize);

    // If shift is held and we are resizing from a corner, retain aspect ratio - needs special handling. We skip this
    // if alt/opt is held - this requires math too big for my brain.
    if (shift && CORNER_ANCHORS.includes(anchor) && !alt) {
      // Fit the bbox to the last aspect ratio
      let fittedWidth = Math.sqrt(width * height * this.$aspectRatioBuffer.get());
      let fittedHeight = fittedWidth / this.$aspectRatioBuffer.get();
      fittedWidth = roundToMultipleMin(fittedWidth, gridSize);
      fittedHeight = roundToMultipleMin(fittedHeight, gridSize);

      // We need to adjust the x and y coords to have the resize occur from the right origin.
      if (anchor === 'top-left') {
        // The transform origin is the bottom-right anchor. Both x and y need to be updated.
        x = x - (fittedWidth - width);
        y = y - (fittedHeight - height);
      }
      if (anchor === 'top-right') {
        // The transform origin is the bottom-left anchor. Only y needs to be updated.
        y = y - (fittedHeight - height);
      }
      if (anchor === 'bottom-left') {
        // The transform origin is the top-right anchor. Only x needs to be updated.
        x = x - (fittedWidth - width);
      }
      // Update the width and height to the fitted dims.
      width = fittedWidth;
      height = fittedHeight;
    }

    const bboxRect = {
      x: Math.round(x),
      y: Math.round(y),
      width,
      height,
    };

    // Update the bboxRect's attrs directly with the new transform, and reset its scale to 1.
    this.konva.proxyRect.setAttrs({ ...bboxRect, scaleX: 1, scaleY: 1 });

    // Update the bbox in internal state.
    this.manager.stateApi.setGenerationBbox(bboxRect);

    // Update the aspect ratio buffer whenever the shift key is not held - this allows for a nice UX where you can start
    // a transform, get the right aspect ratio, then hold shift to lock it in.
    if (!shift) {
      this.$aspectRatioBuffer.set(bboxRect.width / bboxRect.height);
    }
  };

  /**
   * Handles the transformend event on the bbox transformer:
   * - Updates the aspect ratio buffer with the new aspect ratio
   */
  onTransformEnd = () => {
    // Always update the aspect ratio buffer when the transform ends, so if the next transform starts with shift held,
    // we have the correct aspect ratio to start from.
    this.$aspectRatioBuffer.set(this.konva.proxyRect.width() / this.konva.proxyRect.height());
  };

  /**
   * This function is called for each anchor on the transformer. It sets the style of the anchor based on its name.
   * We make the x/y resize anchors little bars.
   */
  anchorStyleFunc = (anchor: Konva.Rect): void => {
    if (anchor.hasName('top-center') || anchor.hasName('bottom-center')) {
      anchor.height(8);
      anchor.offsetY(4);
      anchor.width(30);
      anchor.offsetX(15);
    }
    if (anchor.hasName('middle-left') || anchor.hasName('middle-right')) {
      anchor.height(30);
      anchor.offsetY(15);
      anchor.width(8);
      anchor.offsetX(4);
    }
  };

  /**
   * This function is called for each anchor on the transformer. It sets the drag bounds for the anchor based on the
   * stage's position and the grid size. Care is taken to ensure the anchor snaps to the grid correctly.
   */
  anchorDragBoundFunc = (oldAbsPos: Coordinate, newAbsPos: Coordinate): Coordinate => {
    // This function works with absolute position - that is, a position in "physical" pixels on the screen, as opposed
    // to konva's internal coordinate system.
    const stage = this.konva.transformer.getStage();
    assert(stage, 'Stage must exist');

    // We need to snap the anchors to the grid. If the user is holding ctrl/meta, we use the finer 8px grid.
    const gridSize = this.manager.stateApi.$ctrlKey.get() || this.manager.stateApi.$metaKey.get() ? 8 : 64;
    // Because we are working in absolute coordinates, we need to scale the grid size by the stage scale.
    const scaledGridSize = gridSize * stage.scaleX();
    // To snap the anchor to the grid, we need to calculate an offset from the stage's absolute position.
    const stageAbsPos = stage.getAbsolutePosition();
    // The offset is the remainder of the stage's absolute position divided by the scaled grid size.
    const offsetX = stageAbsPos.x % scaledGridSize;
    const offsetY = stageAbsPos.y % scaledGridSize;
    // Finally, calculate the position by rounding to the grid and adding the offset.
    return {
      x: roundToMultiple(newAbsPos.x, scaledGridSize) + offsetX,
      y: roundToMultiple(newAbsPos.y, scaledGridSize) + offsetY,
    };
  };
}
