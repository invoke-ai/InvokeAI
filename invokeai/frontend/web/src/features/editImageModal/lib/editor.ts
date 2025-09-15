import { $crossOrigin } from 'app/store/nanostores/authToken';
import { TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { objectEntries } from 'tsafe';

/**
 * The position and size of a crop box.
 */
export type CropBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

/**
 * The callbacks supported by the editor.
 */
type EditorCallbacks = {
  onCropStart: Set<() => void>;
  onCropBoxChange: Set<(crop: CropBox) => void>;
  onCropApply: Set<(crop: CropBox) => void>;
  onCropReset: Set<() => void>;
  onCropCancel: Set<() => void>;
  onZoomChange: Set<(zoom: number) => void>;
  onImageLoad: Set<() => void>;
};

type SetElement<T> = T extends Set<infer U> ? U : never;

/**
 * Crop box resize handle names.
 */
type HandleName = 'top-left' | 'top-right' | 'bottom-right' | 'bottom-left' | 'top' | 'right' | 'bottom' | 'left';

/**
 * Crop box guide line names.
 */
type GuideName = 'left' | 'right' | 'top' | 'bottom';

/**
 * All the Konva objects used by the editor, organized by function and approximating the Konva node structures.
 */
type KonvaObjects = {
  stage: Konva.Stage;
  bg: {
    layer: Konva.Layer;
    rect: Konva.Rect;
  };
  image: {
    layer: Konva.Layer;
    image?: Konva.Image;
  };
  crop: {
    layer: Konva.Layer;
    overlay: {
      group: Konva.Group;
      full: Konva.Rect;
      clear: Konva.Rect;
    };
    interaction: {
      group: Konva.Group;
      rect: Konva.Rect;
      handles: Record<HandleName, Konva.Rect>;
      guides: Record<GuideName, Konva.Line>;
    };
  };
};

/**
 * Valid editor output formats.
 */
type OutputFormat = 'canvas' | 'blob' | 'dataURL';

/**
 * Type helper mapping output format name to the actual data type.
 */
type OutputFormatToOutputMap<T extends OutputFormat> = T extends 'canvas'
  ? HTMLCanvasElement
  : T extends 'blob'
    ? Blob
    : T extends 'dataURL'
      ? string
      : never;

/**
 * The editor's configurable parameters.
 */
type EditorConfig = {
  /**
   * The minimum size for the crop box. Applied to both width and height.
   */
  MIN_CROP_DIMENSION: number;

  /**
   * The zoom factor applied when zooming with the mouse wheel. A value of 1.1 means each wheel step zooms in/out by 10%.
   */
  ZOOM_WHEEL_FACTOR: number;

  /**
   * The zoom factor applied when zooming with buttons (e.g. the editor's zoomIn/zoomOut methods). A value of 1.2 means
   * each button press zooms in/out by 20%.
   */
  ZOOM_BUTTON_FACTOR: number;

  /**
   * The size of the crop box resize handles. The handles do not scale with zoom; this is the size they will appear on screen.
   */
  CROP_HANDLE_SIZE: number;

  /**
   * The stroke width of the crop box resize handles. The stroke does not scale with zoom; this is the width it will appear on screen.
   */
  CROP_HANDLE_STROKE_WIDTH: number;

  /**
   * The fill color for the crop box resize handles.
   */
  CROP_HANDLE_FILL: string;

  /**
   * The stroke color for the crop box resize handles.
   */
  CROP_HANDLE_STROKE: string;

  /**
   * The stroke color for the group box guides.
   */
  CROP_GUIDE_STROKE: string;

  /**
   * The stroke width for the crop box guides. The stroke does not scale with zoom; this is the width it will appear on screen.
   */
  CROP_GUIDE_STROKE_WIDTH: number;

  /**
   * The fill color for the crop overlay (the darkened area outside the crop box).
   */
  CROP_OVERLAY_FILL_COLOR: string;

  /**
   * When fitting the image to the container, this padding factor is applied to ensure some space around the image.
   */
  FIT_TO_CONTAINER_PADDING_PCT: number;

  /**
   * When starting a new crop, the initial crop box will be this fraction of the image size.
   */
  DEFAULT_CROP_BOX_SCALE: number;

  /**
   * The minimum zoom (scale) for the stage.
   */
  ZOOM_MIN_PCT: number;

  /**
   * The maximum zoom (scale) for the stage.
   */
  ZOOM_MAX_PCT: number;
};

const DEFAULT_CONFIG: EditorConfig = {
  MIN_CROP_DIMENSION: 64,
  ZOOM_WHEEL_FACTOR: 1.1,
  ZOOM_BUTTON_FACTOR: 1.2,
  CROP_HANDLE_SIZE: 8,
  CROP_HANDLE_STROKE_WIDTH: 1,
  CROP_HANDLE_FILL: 'white',
  CROP_HANDLE_STROKE: 'black',
  CROP_GUIDE_STROKE: 'rgba(255, 255, 255, 0.5)',
  CROP_GUIDE_STROKE_WIDTH: 1,
  CROP_OVERLAY_FILL_COLOR: 'rgba(0, 0, 0, 0.8)',
  FIT_TO_CONTAINER_PADDING_PCT: 0.9,
  DEFAULT_CROP_BOX_SCALE: 0.8,
  ZOOM_MIN_PCT: 0.1,
  ZOOM_MAX_PCT: 10,
};

export class Editor {
  private konva: KonvaObjects | null = null;
  private originalImage: HTMLImageElement | null = null;
  private config: EditorConfig = DEFAULT_CONFIG;

  private aspectRatio: number | null = null;

  private callbacks: EditorCallbacks = {
    onCropApply: new Set(),
    onCropBoxChange: new Set(),
    onCropCancel: new Set(),
    onCropReset: new Set(),
    onCropStart: new Set(),
    onZoomChange: new Set(),
    onImageLoad: new Set(),
  };

  private cropBox: CropBox | null = null;

  // State
  private isCropping = false;
  private isPanning = false;
  private lastPointerPosition: { x: number; y: number } | null = null;
  private isSpacePressed = false;

  private cleanupFunctions: Set<() => void> = new Set();

  /**
   * Initialize the editor inside the given container element.
   * @param container The HTML element to contain the editor. It will be used as the Konva stage container.
   * @param config Optional configuration overrides.
   */
  init = (container: HTMLDivElement, config?: Partial<EditorConfig>) => {
    this.config = { ...this.config, ...config };

    const stage = new Konva.Stage({
      container: container,
      width: container.clientWidth,
      height: container.clientHeight,
    });

    const bg = this.createKonvaBgObjects();
    const image = this.createKonvaImageObjects();
    const crop = this.createKonvaCropObjects();

    stage.add(bg.layer);
    stage.add(image.layer);
    stage.add(crop.layer);

    this.konva = {
      stage,
      bg,
      image,
      crop,
    };

    this.setupListeners();

    if (this.originalImage) {
      this.updateImage();
    }
  };

  /**
   * Create the Konva objects used for the background layer (checkerboard pattern).
   */
  private createKonvaBgObjects = (): KonvaObjects['bg'] => {
    const layer = new Konva.Layer();
    const rect = new Konva.Rect();
    const image = new Image();
    image.onload = () => {
      rect.fillPatternImage(image);
      this.updateKonvaBg();
    };
    image.src = TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL;
    layer.add(rect);

    return {
      layer,
      rect,
    };
  };

  /**
   * Create the Konva objects used for the image layer. Note that the Konva image node is created when an image is loaded.
   */
  private createKonvaImageObjects = (): KonvaObjects['image'] => {
    const layer = new Konva.Layer();
    return {
      layer,
    };
  };

  /**
   * Create the Konva objects used for cropping (overlay and interaction).
   */
  private createKonvaCropObjects = (): KonvaObjects['crop'] => {
    const layer = new Konva.Layer();
    const overlay = this.createKonvaCropOverlayObjects();
    const interaction = this.createKonvaCropInteractionObjects();
    layer.add(overlay.group);
    layer.add(interaction.group);
    return {
      layer,
      overlay,
      interaction,
    };
  };

  /**
   * Create the Konva objects used for the crop overlay (the darkened area outside the crop box).
   *
   * This includes a full rectangle covering the entire image and a rectangle matching the crop box which is used to
   * "cut out" the crop area from the overlay using the 'destination-out' composite operation.
   */
  private createKonvaCropOverlayObjects = (): KonvaObjects['crop']['overlay'] => {
    const group = new Konva.Group();
    const full = new Konva.Rect({
      fill: this.config.CROP_OVERLAY_FILL_COLOR,
    });
    const clear = new Konva.Rect({
      fill: 'black',
      globalCompositeOperation: 'destination-out',
    });
    group.add(full);
    group.add(clear);
    return {
      group,
      full,
      clear,
    };
  };

  /**
   * Create the Konva objects used for crop interaction (the crop box, resize handles, and guides).
   */
  private createKonvaCropInteractionObjects = (): KonvaObjects['crop']['interaction'] => {
    const group = new Konva.Group({ visible: false });

    const rect = this.createKonvaCropInteractionRect();
    const handles = {
      'top-left': this.createKonvaCropInteractionHandle('top-left'),
      'top-right': this.createKonvaCropInteractionHandle('top-right'),
      'bottom-right': this.createKonvaCropInteractionHandle('bottom-right'),
      'bottom-left': this.createKonvaCropInteractionHandle('bottom-left'),
      top: this.createKonvaCropInteractionHandle('top'),
      right: this.createKonvaCropInteractionHandle('right'),
      bottom: this.createKonvaCropInteractionHandle('bottom'),
      left: this.createKonvaCropInteractionHandle('left'),
    };
    const guides = {
      left: this.createKonvaCropInteractionGuide('left'),
      right: this.createKonvaCropInteractionGuide('right'),
      top: this.createKonvaCropInteractionGuide('top'),
      bottom: this.createKonvaCropInteractionGuide('bottom'),
    };

    group.add(rect);

    for (const handle of Object.values(handles)) {
      group.add(handle);
    }
    for (const guide of Object.values(guides)) {
      group.add(guide);
    }

    return {
      group,
      rect,
      handles,
      guides,
    };
  };

  /**
   * Create the Konva rectangle used for crop box interaction (dragging the crop box).
   */
  private createKonvaCropInteractionRect = (): Konva.Rect => {
    const rect = new Konva.Rect({
      stroke: 'white',
      strokeWidth: 1,
      strokeScaleEnabled: false,
      draggable: true,
    });

    // Prevent crop box dragging when panning
    rect.on('dragstart', (e) => {
      if (this.isSpacePressed || this.isPanning) {
        e.target.stopDrag();
        return false;
      }
    });

    // Crop box dragging
    rect.on('dragmove', () => {
      if (!this.konva?.image.image || !this.cropBox) {
        return;
      }

      const imgWidth = this.konva.image.image.width();
      const imgHeight = this.konva.image.image.height();

      // Constrain to image bounds
      const x = Math.max(0, Math.min(rect.x(), imgWidth - rect.width()));
      const y = Math.max(0, Math.min(rect.y(), imgHeight - rect.height()));
      const { width, height } = this.cropBox;

      rect.x(x);
      rect.y(y);

      this.updateCropBox({ x, y, width, height });
    });

    rect.on('mouseenter', () => {
      const stage = this.konva?.stage;
      if (!stage) {
        return;
      }
      if (!this.isSpacePressed) {
        stage.container().style.cursor = 'move';
      }
    });

    rect.on('mouseleave', () => {
      const stage = this.konva?.stage;
      if (!stage) {
        return;
      }
      if (!this.isSpacePressed) {
        stage.container().style.cursor = 'default';
      }
    });

    return rect;
  };

  /**
   * Create a Konva line used as a crop box guide (one of the "rule of thirds" lines).
   */
  private createKonvaCropInteractionGuide = (name: GuideName): Konva.Line => {
    const line = new Konva.Line({
      name,
      stroke: this.config.CROP_GUIDE_STROKE,
      strokeWidth: this.config.CROP_GUIDE_STROKE_WIDTH,
      strokeScaleEnabled: false,
      listening: false,
    });

    return line;
  };

  /**
   * Create a Konva rectangle used as a crop box resize handle.
   */
  private createKonvaCropInteractionHandle = (name: HandleName): Konva.Rect => {
    const rect = new Konva.Rect({
      name,
      x: 0,
      y: 0,
      width: this.config.CROP_HANDLE_SIZE,
      height: this.config.CROP_HANDLE_SIZE,
      fill: this.config.CROP_HANDLE_FILL,
      stroke: this.config.CROP_HANDLE_STROKE,
      strokeWidth: this.config.CROP_HANDLE_STROKE_WIDTH,
      strokeScaleEnabled: true,
      draggable: true,
      hitStrokeWidth: 16,
    });

    // Prevent handle dragging when panning
    rect.on('dragstart', () => {
      const stage = this.konva?.stage;
      if (!stage) {
        return;
      }
      if (stage.isDragging()) {
        rect.stopDrag();
        return false;
      }
    });

    // Set cursor based on handle type
    rect.on('mouseenter', () => {
      const stage = this.konva?.stage;
      if (!stage) {
        return;
      }
      if (!stage.isDragging()) {
        let cursor = 'pointer';
        if (name === 'top-left' || name === 'bottom-right') {
          cursor = 'nwse-resize';
        } else if (name === 'top-right' || name === 'bottom-left') {
          cursor = 'nesw-resize';
        } else if (name === 'top' || name === 'bottom') {
          cursor = 'ns-resize';
        } else if (name === 'left' || name === 'right') {
          cursor = 'ew-resize';
        }
        stage.container().style.cursor = cursor;
      }
    });

    rect.on('mouseleave', () => {
      const stage = this.konva?.stage;
      if (!stage) {
        return;
      }
      if (!stage.isDragging()) {
        stage.container().style.cursor = 'default';
      }
    });

    // Handle dragging
    rect.on('dragmove', () => {
      if (!this.konva) {
        return;
      }

      const { newX, newY, newWidth, newHeight } = this.aspectRatio
        ? this.getNextCropBoxByHandleWithAspectRatio(name, rect)
        : this.getNextCropBoxByHandleFree(name, rect);

      this.updateCropBox({
        x: newX,
        y: newY,
        width: newWidth,
        height: newHeight,
      });
    });

    return rect;
  };

  /**
   * Update (render) the Konva rectangle used for crop box interaction (dragging the crop box).
   */
  private updateKonvaCropInteractionRect = () => {
    if (!this.konva || !this.cropBox) {
      return;
    }
    this.konva.crop.interaction.rect.setAttrs({ ...this.cropBox });
  };

  /**
   * Update (render) the Konva lines used as crop box guides (the "rule of thirds" lines).
   */
  private updateKonvaCropInteractionGuides = () => {
    if (!this.konva || !this.cropBox) {
      return;
    }

    const { x, y, width, height } = this.cropBox;

    const verticalThird = width / 3;
    this.konva.crop.interaction.guides.left.points([x + verticalThird, y, x + verticalThird, y + height]);
    this.konva.crop.interaction.guides.right.points([x + verticalThird * 2, y, x + verticalThird * 2, y + height]);

    const horizontalThird = height / 3;
    this.konva.crop.interaction.guides.top.points([x, y + horizontalThird, x + width, y + horizontalThird]);
    this.konva.crop.interaction.guides.bottom.points([x, y + horizontalThird * 2, x + width, y + horizontalThird * 2]);
  };

  /**
   * Update (render) the Konva rectangles used as crop box resize handles. Only the positions are updated in this
   * method.
   */
  private updateKonvaCropInteractionHandlePositions = () => {
    if (!this.konva || !this.cropBox) {
      return;
    }

    for (const [handleName, handleRect] of objectEntries(this.konva.crop.interaction.handles)) {
      const { x, y, width, height } = this.cropBox;
      const handleSize = handleRect.width();

      let handleX = x;
      let handleY = y;

      if (handleName.includes('right')) {
        handleX += width;
      } else if (!handleName.includes('left')) {
        handleX += width / 2;
      }

      if (handleName.includes('bottom')) {
        handleY += height;
      } else if (!handleName.includes('top')) {
        handleY += height / 2;
      }

      handleRect.x(handleX - handleSize / 2);
      handleRect.y(handleY - handleSize / 2);
    }
  };

  /**
   * Update (render) the Konva rectangles used as crop box resize handles. Only the sizes and stroke widths are updated
   * in this method to maintain a constant screen size regardless of zoom level.
   */
  private updateKonvaCropInteractionHandleScales = () => {
    if (!this.konva) {
      return;
    }

    const scale = this.konva.stage.scaleX();
    const handleSize = this.config.CROP_HANDLE_SIZE / scale;
    const strokeWidth = this.config.CROP_HANDLE_STROKE_WIDTH / scale;

    for (const handle of Object.values(this.konva.crop.interaction.handles)) {
      const currentX = handle.x();
      const currentY = handle.y();
      const oldSize = handle.width();

      // Calculate center position
      const centerX = currentX + oldSize / 2;
      const centerY = currentY + oldSize / 2;

      // Update size and stroke
      handle.width(handleSize);
      handle.height(handleSize);
      handle.strokeWidth(strokeWidth);

      // Reposition to maintain center
      handle.x(centerX - handleSize / 2);
      handle.y(centerY - handleSize / 2);
    }
  };

  /**
   * Update the crop box state and re-render all related Konva objects.
   */
  private updateCropBox = (cropBox: CropBox) => {
    const { x, y, width, height } = cropBox;
    this.cropBox = {
      x: Math.floor(x),
      y: Math.floor(y),
      width: Math.floor(width),
      height: Math.floor(height),
    };
    this.updateKonvaCropOverlay();
    this.updateKonvaCropInteractionRect();
    this.updateKonvaCropInteractionGuides();
    this.updateKonvaCropInteractionHandlePositions();
    this._invokeCallbacks('onCropBoxChange', cropBox);
  };

  /**
   * Update (render) the Konva background objects (the checkerboard pattern).
   */
  private updateKonvaBg = () => {
    if (!this.konva) {
      return;
    }
    const scale = this.konva.stage.scaleX();
    const patternScale = 1 / scale;
    const { x, y } = this.konva.stage.getPosition();
    const { width, height } = this.konva.stage.size();

    this.konva.bg.rect.setAttrs({
      visible: true,
      x: Math.floor(-x / scale),
      y: Math.floor(-y / scale),
      width: Math.ceil(width / scale),
      height: Math.ceil(height / scale),
      fillPatternScaleX: patternScale,
      fillPatternScaleY: patternScale,
    });
  };

  /**
   * Update (render) the Konva crop overlay objects (the darkened area outside the crop box).
   */
  private updateKonvaCropOverlay = () => {
    if (!this.konva?.image.image || !this.cropBox) {
      return;
    }

    // Make the overlay cover the entire image
    this.konva.crop.overlay.full.setAttrs({
      ...this.konva.image.image.getPosition(),
      ...this.konva.image.image.getSize(),
    });

    // Clear the crop area from the overlay
    this.konva.crop.overlay.clear.setAttrs({ ...this.cropBox });
  };

  /**
   * Update (render) the Konva image object when a new image is loaded.
   *
   * This shouldn't be called during normal renders.
   */
  private updateImage = () => {
    if (!this.originalImage || !this.konva) {
      return;
    }

    // Clear existing image
    if (this.konva.image.image) {
      this.konva.image.image.destroy();
      this.konva.image.image = undefined;
    }

    const imageNode = new Konva.Image({
      image: this.originalImage,
      x: 0,
      y: 0,
      width: this.originalImage.width,
      height: this.originalImage.height,
    });

    this.konva.image.image = imageNode;
    this.konva.image.layer.add(imageNode);

    // Center image at 100% zoom
    this.resetView();

    if (this.cropBox) {
      this.updateKonvaCropOverlay();
    }
  };

  /**
   * Calculate the next crop box dimensions when dragging a handle in freeform mode (no aspect ratio).
   *
   * The handle that was dragged determines which edges of the crop box are adjusted.
   *
   * TODO(psyche): Konva's Transformer class can handle this logic. Explore refactoring to use it.
   */
  private getNextCropBoxByHandleFree = (handleName: HandleName, handleRect: Konva.Rect) => {
    if (!this.konva?.image.image || !this.cropBox) {
      throw new Error('Crop box or image not found');
    }

    const imgWidth = this.konva.image.image.width();
    const imgHeight = this.konva.image.image.height();

    let newX = this.cropBox.x;
    let newY = this.cropBox.y;
    let newWidth = this.cropBox.width;
    let newHeight = this.cropBox.height;

    const handleX = handleRect.x() + handleRect.width() / 2;
    const handleY = handleRect.y() + handleRect.height() / 2;

    const minWidth = this.config.MIN_CROP_DIMENSION;
    const minHeight = this.config.MIN_CROP_DIMENSION;

    // Update dimensions based on handle type
    if (handleName.includes('left')) {
      const right = newX + newWidth;
      newX = Math.max(0, Math.min(handleX, right - minWidth));
      newWidth = right - newX;
    }
    if (handleName.includes('right')) {
      newWidth = Math.max(minWidth, Math.min(handleX - newX, imgWidth - newX));
    }
    if (handleName.includes('top')) {
      const bottom = newY + newHeight;
      newY = Math.max(0, Math.min(handleY, bottom - minHeight));
      newHeight = bottom - newY;
    }
    if (handleName.includes('bottom')) {
      newHeight = Math.max(minHeight, Math.min(handleY - newY, imgHeight - newY));
    }

    return { newX, newY, newWidth, newHeight };
  };

  /**
   * Calculate the next crop box dimensions when dragging a handle in fixed aspect ratio mode.
   *
   * The handle that was dragged determines which edges of the crop box are adjusted.
   *
   * TODO(psyche): Konva's Transformer class can handle this logic. Explore refactoring to use it.
   */
  private getNextCropBoxByHandleWithAspectRatio = (handleName: HandleName, handleRect: Konva.Rect) => {
    if (!this.konva?.image.image || !this.aspectRatio || !this.cropBox) {
      throw new Error('Crop box, image, or aspect ratio not found');
    }
    const imgWidth = this.konva.image.image.width();
    const imgHeight = this.konva.image.image.height();
    const ratio = this.aspectRatio;

    const handleX = handleRect.x() + handleRect.width() / 2;
    const handleY = handleRect.y() + handleRect.height() / 2;

    const minWidth = this.config.MIN_CROP_DIMENSION;
    const minHeight = this.config.MIN_CROP_DIMENSION;

    // Early boundary check for aspect ratio mode
    const atLeftEdge = this.cropBox.x <= 0;
    const atRightEdge = this.cropBox.x + this.cropBox.width >= imgWidth;
    const atTopEdge = this.cropBox.y <= 0;
    const atBottomEdge = this.cropBox.y + this.cropBox.height >= imgHeight;

    if (
      (handleName === 'left' && atLeftEdge && handleX < this.cropBox.x) ||
      (handleName === 'right' && atRightEdge && handleX > this.cropBox.x + this.cropBox.width) ||
      (handleName === 'top' && atTopEdge && handleY < this.cropBox.y) ||
      (handleName === 'bottom' && atBottomEdge && handleY > this.cropBox.y + this.cropBox.height)
    ) {
      return {
        newX: this.cropBox.x,
        newY: this.cropBox.y,
        newWidth: this.cropBox.width,
        newHeight: this.cropBox.height,
      };
    }

    const {
      newX: freeX,
      newY: freeY,
      newWidth: freeWidth,
      newHeight: freeHeight,
    } = this.getNextCropBoxByHandleFree(handleName, handleRect);

    let newX = freeX;
    let newY = freeY;
    let newWidth = freeWidth;
    let newHeight = freeHeight;

    const oldX = this.cropBox.x;
    const oldY = this.cropBox.y;
    const oldWidth = this.cropBox.width;
    const oldHeight = this.cropBox.height;

    // Define anchor points (opposite of the handle being dragged)
    let anchorX = oldX;
    let anchorY = oldY;

    if (handleName.includes('right')) {
      anchorX = oldX; // Left edge is anchor
    } else if (handleName.includes('left')) {
      anchorX = oldX + oldWidth; // Right edge is anchor
    } else {
      anchorX = oldX + oldWidth / 2; // Center X is anchor for top/bottom
    }

    if (handleName.includes('bottom')) {
      anchorY = oldY; // Top edge is anchor
    } else if (handleName.includes('top')) {
      anchorY = oldY + oldHeight; // Bottom edge is anchor
    } else {
      anchorY = oldY + oldHeight / 2; // Center Y is anchor for left/right
    }

    // Calculate new dimensions maintaining aspect ratio
    if (handleName === 'left' || handleName === 'right' || handleName === 'top' || handleName === 'bottom') {
      if (handleName === 'left' || handleName === 'right') {
        newHeight = newWidth / ratio;
        newY = anchorY - newHeight / 2;
      } else {
        // top or bottom
        newWidth = newHeight * ratio;
        newX = anchorX - newWidth / 2;
      }
    } else {
      // Corner handles
      const mouseDistanceFromAnchorX = Math.abs(handleX - anchorX);
      const mouseDistanceFromAnchorY = Math.abs(handleY - anchorY);

      let maxPossibleWidth = handleName.includes('left') ? anchorX : imgWidth - anchorX;
      let maxPossibleHeight = handleName.includes('top') ? anchorY : imgHeight - anchorY;

      const constrainedMouseDistanceX = Math.min(mouseDistanceFromAnchorX, maxPossibleWidth);
      const constrainedMouseDistanceY = Math.min(mouseDistanceFromAnchorY, maxPossibleHeight);

      if (constrainedMouseDistanceX / ratio > constrainedMouseDistanceY) {
        newWidth = constrainedMouseDistanceX;
        newHeight = newWidth / ratio;
        if (newHeight > maxPossibleHeight) {
          newHeight = maxPossibleHeight;
          newWidth = newHeight * ratio;
        }
      } else {
        newHeight = constrainedMouseDistanceY;
        newWidth = newHeight * ratio;
        if (newWidth > maxPossibleWidth) {
          newWidth = maxPossibleWidth;
          newHeight = newWidth / ratio;
        }
      }

      newX = handleName.includes('left') ? anchorX - newWidth : anchorX;
      newY = handleName.includes('top') ? anchorY - newHeight : anchorY;
    }

    // Boundary checks and adjustments
    if (newX < 0) {
      newX = 0;
      newWidth = oldX + oldWidth;
      newHeight = newWidth / ratio;
      newY = handleName.includes('top') ? oldY + oldHeight - newHeight : oldY;
    }
    if (newY < 0) {
      newY = 0;
      newHeight = oldY + oldHeight;
      newWidth = newHeight * ratio;
      newX = handleName.includes('left') ? oldX + oldWidth - newWidth : oldX;
    }
    if (newX + newWidth > imgWidth) {
      newWidth = imgWidth - newX;
      newHeight = newWidth / ratio;
      newY = handleName.includes('top') ? oldY + oldHeight - newHeight : oldY;
    }
    if (newY + newHeight > imgHeight) {
      newHeight = imgHeight - newY;
      newWidth = newHeight * ratio;
      newX = handleName.includes('left') ? oldX + oldWidth - newWidth : oldX;
    }

    // Final check for minimum sizes
    if (newWidth < minWidth || newHeight < minHeight) {
      if (minWidth / ratio > minHeight) {
        newWidth = minWidth;
        newHeight = newWidth / ratio;
      } else {
        newHeight = minHeight;
        newWidth = newHeight * ratio;
      }
      newX = handleName.includes('left') ? anchorX - newWidth : anchorX;
      newY = handleName.includes('top') ? anchorY - newHeight : anchorY;
    }

    return { newX, newY, newWidth, newHeight };
  };

  //#region Event Handling

  /**
   * Set up event listeners for Konva stage (pointer) and window events (keyboard).
   */
  private setupListeners = () => {
    if (!this.konva) {
      return;
    }

    const stage = this.konva.stage;

    stage.on('wheel', this.onWheel);
    stage.on('contextmenu', this.onContextMenu);
    stage.on('pointerdown', this.onPointerDown);
    stage.on('pointerup', this.onPointerUp);
    stage.on('pointermove', this.onPointerMove);

    this.cleanupFunctions.add(() => {
      stage.off('wheel', this.onWheel);
      stage.off('contextmenu', this.onContextMenu);
      stage.off('pointerdown', this.onPointerDown);
      stage.off('pointerup', this.onPointerUp);
      stage.off('pointermove', this.onPointerMove);
    });

    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);

    this.cleanupFunctions.add(() => {
      window.removeEventListener('keydown', this.onKeyDown);
      window.removeEventListener('keyup', this.onKeyUp);
    });
  };

  /**
   * Handle keydown events.
   * - Space: Enable panning mode.
   */
  private onKeyDown = (e: KeyboardEvent) => {
    if (!this.konva?.stage) {
      return;
    }
    if (e.code === 'Space' && !this.isSpacePressed) {
      e.preventDefault();
      this.isSpacePressed = true;
      this.konva.stage.container().style.cursor = 'grab';
    }
  };

  /**
   * Handle keyup events.
   * - Space: Disable panning mode.
   */
  private onKeyUp = (e: KeyboardEvent) => {
    if (!this.konva?.stage) {
      return;
    }
    if (e.code === 'Space') {
      e.preventDefault();
      this.isSpacePressed = false;
      this.isPanning = false;
      // Revert cursor to default; mouseenter events will set it correctly if over an interactive element.
      this.konva.stage.container().style.cursor = 'default';
    }
  };

  /**
   * Handle mouse wheel events for zooming in/out.
   * - Zoom is centered on the mouse pointer position and constrained to min/max levels.
   * - The crop box handles are rescaled to maintain a constant screen size.
   * - The background pattern is rescalted to maintain a constant screen size.
   */
  private onWheel = (e: KonvaEventObject<WheelEvent>) => {
    if (!this.konva?.stage) {
      return;
    }
    e.evt.preventDefault();

    const oldScale = this.konva.stage.scaleX();
    const pointer = this.konva.stage.getPointerPosition();

    if (!pointer) {
      return;
    }

    const mousePointTo = {
      x: (pointer.x - this.konva.stage.x()) / oldScale,
      y: (pointer.y - this.konva.stage.y()) / oldScale,
    };

    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const scaleBy = this.config.ZOOM_WHEEL_FACTOR;
    let newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;

    // Apply zoom limits
    newScale = Math.max(this.config.ZOOM_MIN_PCT, Math.min(this.config.ZOOM_MAX_PCT, newScale));

    this.konva.stage.scale({ x: newScale, y: newScale });

    const newPos = {
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    };
    this.konva.stage.position(newPos);

    // Update handle scaling to maintain constant screen size
    this.updateKonvaCropInteractionHandleScales();
    this.updateKonvaBg();
    this._invokeCallbacks('onZoomChange', newScale);
  };

  /**
   * Handle pointer down events to initiate panning mode if spacebar is pressed or middle mouse button is used.
   * - Stops any active drags on crop elements to prevent conflicts.
   */
  private onPointerDown = (e: KonvaEventObject<PointerEvent>) => {
    if (!this.konva?.stage) {
      return;
    }
    if (this.isSpacePressed || e.evt.button === 1) {
      e.evt.preventDefault();
      e.evt.stopPropagation();
      this.isPanning = true;
      this.lastPointerPosition = this.konva.stage.getPointerPosition();
      this.konva.stage.container().style.cursor = 'grabbing';

      // Stop any active drags on crop elements
      if (this.konva.crop) {
        if (this.konva.crop.interaction.rect.isDragging()) {
          this.konva.crop.interaction.rect.stopDrag();
        }
        for (const handle of Object.values(this.konva.crop.interaction.handles)) {
          if (handle.isDragging()) {
            handle.stopDrag();
          }
        }
      }
    }
  };

  /**
   * Handle pointer move events to pan the image when in panning mode.
   */
  private onPointerMove = (_: KonvaEventObject<PointerEvent>) => {
    if (!this.konva?.stage) {
      return;
    }
    if (!this.isPanning || !this.lastPointerPosition) {
      return;
    }

    const pointer = this.konva.stage.getPointerPosition();
    if (!pointer) {
      return;
    }

    const dx = pointer.x - this.lastPointerPosition.x;
    const dy = pointer.y - this.lastPointerPosition.y;

    this.konva.stage.x(this.konva.stage.x() + dx);
    this.konva.stage.y(this.konva.stage.y() + dy);

    this.updateKonvaBg();

    this.lastPointerPosition = pointer;
  };

  /**
   * Handle pointer up events to exit panning mode.
   */
  private onPointerUp = (_: KonvaEventObject<PointerEvent>) => {
    if (!this.konva?.stage) {
      return;
    }
    if (this.isPanning) {
      this.isPanning = false;
      this.konva.stage.container().style.cursor = this.isSpacePressed ? 'grab' : 'default';
    }
  };

  /**
   * Handle context menu events to prevent the default browser context menu from appearing on right-click.
   */
  private onContextMenu = (e: KonvaEventObject<MouseEvent>) => {
    e.evt.preventDefault();
  };
  //#region Event Handling

  //#region Public API

  /**
   * Load an image from a URL or data URL.
   * @param src The image source URL or data URL.
   * @returns A promise that resolves when the image is loaded or rejects on error.
   */
  loadImage = (src: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      const img = new Image();

      img.crossOrigin = $crossOrigin.get();

      img.onload = () => {
        this.originalImage = img;
        this.updateImage();
        this._invokeCallbacks('onImageLoad');
        resolve();
      };

      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };

      img.src = src;
    });
  };

  /**
   * Start a cropping session with an optional initial crop box.
   * @param initialCrop Optional initial crop box to use. If not provided, uses the current crop box or a default centered box.
   */
  startCrop = (initialCrop?: CropBox) => {
    if (!this.konva?.image.image || this.isCropping) {
      return;
    }

    // Calculate initial crop dimensions
    let cropX: number;
    let cropY: number;
    let cropWidth: number;
    let cropHeight: number;

    if (initialCrop) {
      // User provided initial crop
      cropX = initialCrop.x;
      cropY = initialCrop.y;
      cropWidth = initialCrop.width;
      cropHeight = initialCrop.height;
    } else if (this.cropBox) {
      // Use the current crop as starting point
      cropX = this.cropBox.x;
      cropY = this.cropBox.y;
      cropWidth = this.cropBox.width;
      cropHeight = this.cropBox.height;
    } else {
      // Create default crop box (centered, 80% of image)
      const imgWidth = this.konva.image.image.width();
      const imgHeight = this.konva.image.image.height();
      cropWidth = imgWidth * this.config.DEFAULT_CROP_BOX_SCALE;
      cropHeight = imgHeight * this.config.DEFAULT_CROP_BOX_SCALE;
      cropX = (imgWidth - cropWidth) / 2;
      cropY = (imgHeight - cropHeight) / 2;
    }

    this.updateCropBox({
      x: cropX,
      y: cropY,
      width: cropWidth,
      height: cropHeight,
    });
    this.isCropping = true;
    this.konva.crop.interaction.group.visible(true);

    this._invokeCallbacks('onCropStart');
  };

  /**
   * Cancel the current cropping session and hide crop UI.
   */
  cancelCrop = () => {
    if (!this.isCropping || !this.konva) {
      return;
    }
    this.isCropping = false;
    this.konva.crop.interaction.group.visible(false);
    this._invokeCallbacks('onCropCancel');
  };

  /**
   * Apply the current crop box and exit cropping mode.
   */
  applyCrop = () => {
    if (!this.isCropping || !this.cropBox || !this.konva) {
      return;
    }

    this.isCropping = false;
    this.konva.crop.interaction.group.visible(false);
    this._invokeCallbacks('onCropApply', this.cropBox);
  };

  /**
   * Reset the crop box to encompass the entire image.
   */
  resetCrop = () => {
    if (this.konva?.image.image) {
      this.updateCropBox({
        x: 0,
        y: 0,
        ...this.konva.image.image.size(),
      });
    }
    this._invokeCallbacks('onCropReset');
  };

  /**
   * Export the current image with the current crop applied, in the specified format.
   *
   * If there is no crop box, the full image is exported.
   *
   * @param format The output format: 'canvas', 'blob', or 'dataURL'. Defaults to 'blob'.
   * @returns A promise that resolves with the exported image in the requested format.
   */
  exportImage = <T extends OutputFormat>(
    format: T = 'blob' as T,
    options?: { withCropOverlay?: boolean }
  ): Promise<OutputFormatToOutputMap<T>> => {
    const { withCropOverlay } = { withCropOverlay: false, ...options };
    return new Promise((resolve, reject) => {
      if (!this.originalImage) {
        throw new Error('No image loaded');
      }

      // Create temporary canvas
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }

      try {
        if (this.cropBox) {
          if (!withCropOverlay) {
            // Draw the cropped image
            canvas.width = this.cropBox.width;
            canvas.height = this.cropBox.height;

            ctx.drawImage(
              this.originalImage,
              this.cropBox.x,
              this.cropBox.y,
              this.cropBox.width,
              this.cropBox.height,
              0,
              0,
              this.cropBox.width,
              this.cropBox.height
            );
          } else {
            // Draw the full image with dark overlay and clear crop area
            canvas.width = this.originalImage.width;
            canvas.height = this.originalImage.height;

            ctx.drawImage(this.originalImage, 0, 0);

            // We need a new canvas for the overlay to avoid messing up the original image when clearing the crop area
            const overlayCanvas = document.createElement('canvas');
            overlayCanvas.width = this.originalImage.width;
            overlayCanvas.height = this.originalImage.height;

            const overlayCtx = overlayCanvas.getContext('2d');
            if (!overlayCtx) {
              throw new Error('Failed to get canvas context');
            }

            overlayCtx.fillStyle = this.config.CROP_OVERLAY_FILL_COLOR;
            overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            overlayCtx.clearRect(this.cropBox.x, this.cropBox.y, this.cropBox.width, this.cropBox.height);

            ctx.globalCompositeOperation = 'multiply';
            ctx.drawImage(overlayCanvas, 0, 0);

            overlayCanvas.remove();
          }
        } else {
          canvas.width = this.originalImage.width;
          canvas.height = this.originalImage.height;
          ctx.drawImage(this.originalImage, 0, 0);
        }

        if (format === 'canvas') {
          resolve(canvas as OutputFormatToOutputMap<T>);
        } else if (format === 'dataURL') {
          try {
            resolve(canvas.toDataURL('image/png') as OutputFormatToOutputMap<T>);
          } catch (error) {
            reject(error);
          }
        } else {
          try {
            canvas.toBlob((blob) => {
              if (blob) {
                resolve(blob as OutputFormatToOutputMap<T>);
              } else {
                reject(new Error('Failed to create blob'));
              }
            }, 'image/png');
          } catch (error) {
            reject(error);
          }
        }
      } catch (error) {
        reject(error);
      }
    });
  };

  /**
   * Set the zoom level, optionally centered on a specific point.
   * @param scale The target zoom scale (1 = 100%).
   * @param point Optional point to center the zoom on, in stage coordinates. Defaults to center of viewport.
   */
  setZoom = (scale: number, point?: { x: number; y: number }) => {
    if (!this.konva) {
      return;
    }

    scale = Math.max(this.config.ZOOM_MIN_PCT, Math.min(this.config.ZOOM_MAX_PCT, scale));

    // If no point provided, use center of viewport
    if (!point && this.konva.image) {
      const containerWidth = this.konva.stage.width();
      const containerHeight = this.konva.stage.height();
      point = {
        x: containerWidth / 2,
        y: containerHeight / 2,
      };
    }

    if (point) {
      const oldScale = this.konva.stage.scaleX();
      const mousePointTo = {
        x: (point.x - this.konva.stage.x()) / oldScale,
        y: (point.y - this.konva.stage.y()) / oldScale,
      };

      this.konva.stage.scale({ x: scale, y: scale });

      const newPos = {
        x: point.x - mousePointTo.x * scale,
        y: point.y - mousePointTo.y * scale,
      };
      this.konva.stage.position(newPos);
    } else {
      this.konva.stage.scale({ x: scale, y: scale });
    }

    // Update handle scaling
    this.updateKonvaCropInteractionHandleScales();

    this.updateKonvaBg();

    this._invokeCallbacks('onZoomChange', scale);
  };

  /**
   * Get the current zoom level (1 = 100%).
   */
  getZoom = (): number => {
    return this.konva?.stage.scaleX() || 1;
  };

  /**
   * Zoom in/out by a fixed factor, optionally centered on a specific point.
   * @param point Optional point to center the zoom on, in stage coordinates. Defaults to center of viewport.
   */
  zoomIn = (point?: { x: number; y: number }) => {
    const currentZoom = this.getZoom();
    this.setZoom(currentZoom * this.config.ZOOM_BUTTON_FACTOR, point);
  };

  /**
   * Zoom out by a fixed factor, optionally centered on a specific point.
   * @param point Optional point to center the zoom on, in stage coordinates. Defaults to center of viewport.
   */
  zoomOut = (point?: { x: number; y: number }) => {
    const currentZoom = this.getZoom();
    this.setZoom(currentZoom / this.config.ZOOM_BUTTON_FACTOR, point);
  };

  /**
   * Reset the view to 100% zoom and center the image in the container.
   */
  resetView = () => {
    if (!this.konva?.image.image) {
      return;
    }

    this.konva.stage.scale({ x: 1, y: 1 });

    // Center the image
    const containerWidth = this.konva.stage.width();
    const containerHeight = this.konva.stage.height();
    const imageWidth = this.konva.image.image.width();
    const imageHeight = this.konva.image.image.height();

    this.konva.stage.position({
      x: (containerWidth - imageWidth) / 2,
      y: (containerHeight - imageHeight) / 2,
    });

    // Update handle scaling
    this.updateKonvaCropInteractionHandleScales();

    this.updateKonvaBg();

    this._invokeCallbacks('onZoomChange', 1);
  };

  /**
   * Scale the image to fit within the container while maintaining aspect ratio.
   * Adds padding to ensure the image isn't flush against container edges.
   */
  fitToContainer = () => {
    if (!this.konva?.image?.image) {
      return;
    }

    const containerWidth = this.konva.stage.width();
    const containerHeight = this.konva.stage.height();
    const imageWidth = this.konva.image.image.width();
    const imageHeight = this.konva.image.image.height();

    const scale =
      Math.min(containerWidth / imageWidth, containerHeight / imageHeight) * this.config.FIT_TO_CONTAINER_PADDING_PCT;

    this.konva.stage.scale({ x: scale, y: scale });

    // Center the image
    const scaledWidth = imageWidth * scale;
    const scaledHeight = imageHeight * scale;

    this.konva.stage.position({
      x: (containerWidth - scaledWidth) / 2,
      y: (containerHeight - scaledHeight) / 2,
    });

    // Update handle scaling
    this.updateKonvaCropInteractionHandleScales();

    this.updateKonvaBg();

    this._invokeCallbacks('onZoomChange', scale);
  };

  /**
   * Set or update event callbacks.
   * @param callbacks The callbacks to set or update.
   * @param replace If true, replaces all existing callbacks. If false, merges with existing callbacks. Default is false.
   */
  setCallbacks = (callbacks: EditorCallbacks, replace = false) => {
    if (replace) {
      this.callbacks = callbacks;
    } else {
      this.callbacks = { ...this.callbacks, ...callbacks };
    }
  };

  /**
   * Set or update the crop aspect ratio constraint.
   * @param ratio The desired aspect ratio (width / height) or null to remove the constraint.
   *
   * If setting a new aspect ratio, the crop box is adjusted to maintain its area while fitting within image bounds.
   * Minimum size constraints are applied as needed.
   */
  setCropAspectRatio = (ratio: number | null) => {
    this.aspectRatio = ratio;

    if (!this.konva?.image.image || !this.cropBox) {
      return;
    }

    const currentWidth = this.cropBox.width;
    const currentHeight = this.cropBox.height;
    const currentArea = currentWidth * currentHeight;

    if (ratio === null) {
      // Just removed the aspect ratio constraint, no need to adjust
      return;
    }

    // Calculate new dimensions maintaining the same area
    // area = width * height
    // ratio = width / height
    // So: area = width * (width / ratio)
    // Therefore: width = sqrt(area * ratio)
    let newWidth = Math.sqrt(currentArea * ratio);
    let newHeight = newWidth / ratio;

    // Get image bounds
    const imgWidth = this.konva.image.image.width();
    const imgHeight = this.konva.image.image.height();

    // Check if the new dimensions would exceed image bounds
    if (newWidth > imgWidth || newHeight > imgHeight) {
      // Scale down to fit within image bounds while maintaining ratio
      const scaleX = imgWidth / newWidth;
      const scaleY = imgHeight / newHeight;
      const scale = Math.min(scaleX, scaleY);
      newWidth *= scale;
      newHeight *= scale;
    }

    // Apply minimum size constraints
    const minWidth = this.config.MIN_CROP_DIMENSION;
    const minHeight = this.config.MIN_CROP_DIMENSION;

    if (newWidth < minWidth) {
      newWidth = minWidth;
      newHeight = newWidth / ratio;
    }
    if (newHeight < minHeight) {
      newHeight = minHeight;
      newWidth = newHeight * ratio;
    }

    // Center the new crop box at the same position as the old one
    const currentCenterX = this.cropBox.x + currentWidth / 2;
    const currentCenterY = this.cropBox.y + currentHeight / 2;

    let newX = currentCenterX - newWidth / 2;
    let newY = currentCenterY - newHeight / 2;

    // Ensure the crop box stays within image bounds
    newX = Math.max(0, Math.min(newX, imgWidth - newWidth));
    newY = Math.max(0, Math.min(newY, imgHeight - newHeight));

    this.updateCropBox({
      x: newX,
      y: newY,
      width: newWidth,
      height: newHeight,
    });
  };

  setCropBox = (box: CropBox) => {
    this.updateCropBox(box);
  };

  /**
   * Get the current crop aspect ratio constraint.
   * @returns The current aspect ratio (width / height) or null if no constraint is set.
   */
  getCropAspectRatio = (): number | null => {
    return this.aspectRatio;
  };

  /**
   * Helper to build a callback registrar function for a specific event name.
   * @param name The callback event name.
   */
  _buildCallbackRegistrar = <T extends keyof EditorCallbacks>(name: T) => {
    return (cb: SetElement<EditorCallbacks[T]>): (() => void) => {
      (this.callbacks[name] as Set<typeof cb>).add(cb);
      return () => {
        (this.callbacks[name] as Set<typeof cb>).delete(cb);
      };
    };
  };

  /**
   * Invoke all callbacks registered for a specific event.
   * @param name The callback event name.
   * @param args The arguments to pass to each callback.
   */
  private _invokeCallbacks = <T extends keyof EditorCallbacks>(
    name: T,
    ...args: EditorCallbacks[T] extends Set<(...args: infer P) => void> ? P : never
  ): void => {
    const callbacks = this.callbacks[name];
    if (callbacks && callbacks.size > 0) {
      callbacks.forEach((cb) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (cb as (...args: any[]) => void)(...args);
      });
    }
  };

  /**
   * Register a callback for when the crop is applied.
   */
  onCropApply = this._buildCallbackRegistrar('onCropApply');
  /**
   * Register a callback for when the crop is canceled.
   */
  onCropCancel = this._buildCallbackRegistrar('onCropCancel');
  /**
   * Register a callback for when the crop is reset.
   */
  onCropReset = this._buildCallbackRegistrar('onCropReset');
  /**
   * Register a callback for when cropping starts.
   */
  onCropStart = this._buildCallbackRegistrar('onCropStart');
  /**
   * Register a callback for when the crop box changes (moved or resized).
   */
  onCropBoxChange = this._buildCallbackRegistrar('onCropBoxChange');
  /**
   * Register a callback for when a new image is loaded.
   */
  onImageLoad = this._buildCallbackRegistrar('onImageLoad');
  /**
   * Register a callback for when the zoom level changes.
   */
  onZoomChange = this._buildCallbackRegistrar('onZoomChange');

  /**
   * Resize the editor container and adjust the Konva stage accordingly.
   *
   * Use this method when the container size changes (e.g., window resize) to ensure the canvas fits properly.
   *
   * @param width The new container width in pixels.
   * @param height The new container height in pixels.
   */
  resize = (width: number, height: number) => {
    if (!this.konva) {
      return;
    }

    this.konva.stage.width(width);
    this.konva.stage.height(height);

    this.updateKonvaBg();
  };

  /**
   * Destroy the editor instance, cleaning up all resources and event listeners.
   *
   * After calling this method, the instance should not be used again.
   */
  destroy = () => {
    for (const cleanup of this.cleanupFunctions) {
      cleanup();
    }

    // Cancel any ongoing crop operation
    if (this.isCropping) {
      this.cancelCrop();
    }

    this.konva?.stage.destroy();

    // Clear all references
    this.konva = null;
    this.originalImage = null;
    this.cropBox = null;
    for (const set of Object.values(this.callbacks)) {
      set.clear();
    }
  };
  //#endregion Public API
}
