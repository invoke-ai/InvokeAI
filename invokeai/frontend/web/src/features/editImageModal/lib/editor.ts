import { $crossOrigin } from 'app/store/nanostores/authToken';
import { TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import { objectEntries } from 'tsafe';

type CropConstraints = {
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  aspectRatio?: number;
};

export type CropBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type EditorCallbacks = {
  onCropStart?: () => void;
  onCropBoxChange?: (crop: CropBox) => void;
  onCropApply?: (crop: CropBox) => void;
  onCropReset?: () => void;
  onCropCancel?: () => void;
  onZoomChange?: (zoom: number) => void;
  onImageLoad?: () => void;
};

type HandleName = 'top-left' | 'top-right' | 'bottom-right' | 'bottom-left' | 'top' | 'right' | 'bottom' | 'left';

type GuideName = 'left' | 'right' | 'top' | 'bottom';

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

type OutputFormat = 'canvas' | 'blob' | 'dataURL';

type OutputFormatToOutputMap<T extends OutputFormat> = T extends 'canvas'
  ? HTMLCanvasElement
  : T extends 'blob'
    ? Blob
    : T extends 'dataURL'
      ? string
      : never;

export class Editor {
  private konva: KonvaObjects | null = null;
  private originalImage: HTMLImageElement | null = null;
  private isCropping = false;

  // Constants
  private readonly MIN_CROP_DIMENSION = 64;
  private readonly ZOOM_WHEEL_FACTOR = 1.1;
  private readonly ZOOM_BUTTON_FACTOR = 1.2;
  private readonly CROP_HANDLE_SIZE = 8;
  private readonly CROP_HANDLE_STROKE_WIDTH = 1;
  private readonly CROP_GUIDE_STROKE = 'rgba(255, 255, 255, 0.5)';
  private readonly CROP_GUIDE_STROKE_WIDTH = 1;
  private readonly CROP_HANDLE_FILL = 'white';
  private readonly CROP_HANDLE_STROKE = 'black';
  private readonly FIT_TO_CONTAINER_PADDING = 0.9;
  private readonly DEFAULT_CROP_BOX_SCALE = 0.8;

  // Configuration
  private readonly ZOOM_MIN = 0.1;
  private readonly ZOOM_MAX = 10;

  private cropConstraints: CropConstraints = {
    minWidth: this.MIN_CROP_DIMENSION,
    minHeight: this.MIN_CROP_DIMENSION,
  };
  private callbacks: EditorCallbacks = {};
  private cropBox: CropBox | null = null;

  // State
  private isPanning = false;
  private lastPointerPosition: { x: number; y: number } | null = null;
  private isSpacePressed = false;

  subscriptions: Set<() => void> = new Set();

  init = (container: HTMLDivElement) => {
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

    this.setupStageEvents();
  };

  createKonvaBgObjects = (): KonvaObjects['bg'] => {
    const layer = new Konva.Layer();
    const rect = new Konva.Rect();
    const image = new Image();
    image.onload = () => {
      rect.fillPatternImage(image);
      this.renderBg();
    };
    image.src = TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL;
    layer.add(rect);

    return {
      layer,
      rect,
    };
  };

  createKonvaImageObjects = (): KonvaObjects['image'] => {
    const layer = new Konva.Layer();
    return {
      layer,
    };
  };

  createKonvaCropObjects = (): KonvaObjects['crop'] => {
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
  createKonvaCropOverlayObjects = (): KonvaObjects['crop']['overlay'] => {
    const group = new Konva.Group();
    const full = new Konva.Rect({
      fill: 'black',
      opacity: 0.7,
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

  createKonvaCropInteractionObjects = (): KonvaObjects['crop']['interaction'] => {
    const group = new Konva.Group();

    const rect = this.createCropInteractionRect();
    const handles = {
      'top-left': this.createHandle('top-left'),
      'top-right': this.createHandle('top-right'),
      'bottom-right': this.createHandle('bottom-right'),
      'bottom-left': this.createHandle('bottom-left'),
      top: this.createHandle('top'),
      right: this.createHandle('right'),
      bottom: this.createHandle('bottom'),
      left: this.createHandle('left'),
    };
    const guides = {
      left: this.createGuide('left'),
      right: this.createGuide('right'),
      top: this.createGuide('top'),
      bottom: this.createGuide('bottom'),
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

  createCropInteractionRect = (): Konva.Rect => {
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

      rect.x(x);
      rect.y(y);

      this.updateCropBox({
        ...this.cropBox,
        x,
        y,
      });
    });

    // Cursor styles
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

  updateCropInteractionRect = () => {
    if (!this.konva || !this.cropBox) {
      return;
    }
    this.konva.crop.interaction.rect.setAttrs({ ...this.cropBox });
  };

  createGuide = (name: GuideName): Konva.Line => {
    const line = new Konva.Line({
      name,
      stroke: this.CROP_GUIDE_STROKE,
      strokeWidth: this.CROP_GUIDE_STROKE_WIDTH,
      strokeScaleEnabled: false,
      listening: false,
    });

    return line;
  };

  updateCropGuides = () => {
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

  createHandle = (name: HandleName): Konva.Rect => {
    const rect = new Konva.Rect({
      name,
      x: 0,
      y: 0,
      width: this.CROP_HANDLE_SIZE,
      height: this.CROP_HANDLE_SIZE,
      fill: this.CROP_HANDLE_FILL,
      stroke: this.CROP_HANDLE_STROKE,
      strokeWidth: this.CROP_HANDLE_STROKE_WIDTH,
      strokeScaleEnabled: true,
      draggable: true,
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
      this.resizeCropBox(name, rect);
    });

    return rect;
  };

  updateCropBox = (cropBox: CropBox) => {
    this.cropBox = cropBox;
    this.updateCropInteractionRect();
    this.updateCropOverlay();
    this.updateCropGuides();
    this.updateHandlePositions();
    this.callbacks.onCropBoxChange?.(cropBox);
  };

  renderBg = () => {
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

  private setupStageEvents = () => {
    if (!this.konva) {
      return;
    }
    const stage = this.konva.stage;

    stage.container().addEventListener('wheel', this.onWheel, { passive: false });
    this.subscriptions.add(() => {
      stage.container().removeEventListener('wheel', this.onWheel);
    });
    stage.container().addEventListener('contextmenu', this.onContextMenu);
    this.subscriptions.add(() => {
      stage.container().removeEventListener('contextmenu', this.onContextMenu);
    });

    stage.on('pointerdown', this.onPointerDown);
    this.subscriptions.add(() => {
      stage.off('pointerdown', this.onPointerDown);
    });
    stage.on('pointerup', this.onPointerUp);
    this.subscriptions.add(() => {
      stage.off('pointerup', this.onPointerUp);
    });
    stage.on('pointermove', this.onPointerMove);
    this.subscriptions.add(() => {
      stage.off('pointermove', this.onPointerMove);
    });

    window.addEventListener('keydown', this.onKeyDown);
    this.subscriptions.add(() => {
      window.removeEventListener('keydown', this.onKeyDown);
    });

    window.addEventListener('keyup', this.onKeyUp);
    this.subscriptions.add(() => {
      window.removeEventListener('keyup', this.onKeyUp);
    });
  };

  // Track Space key press
  onKeyDown = (e: KeyboardEvent) => {
    if (!this.konva?.stage) {
      return;
    }
    if (e.code === 'Space' && !this.isSpacePressed) {
      e.preventDefault();
      this.isSpacePressed = true;
      this.konva.stage.container().style.cursor = 'grab';
    }
  };

  // Zoom with mouse wheel
  onWheel = (e: WheelEvent) => {
    if (!this.konva?.stage) {
      return;
    }
    e.preventDefault();

    const oldScale = this.konva.stage.scaleX();
    const pointer = this.konva.stage.getPointerPosition();

    if (!pointer) {
      return;
    }

    const mousePointTo = {
      x: (pointer.x - this.konva.stage.x()) / oldScale,
      y: (pointer.y - this.konva.stage.y()) / oldScale,
    };

    const direction = e.deltaY > 0 ? -1 : 1;
    const scaleBy = this.ZOOM_WHEEL_FACTOR;
    let newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;

    // Apply zoom limits
    newScale = Math.max(this.ZOOM_MIN, Math.min(this.ZOOM_MAX, newScale));

    this.konva.stage.scale({ x: newScale, y: newScale });

    const newPos = {
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    };
    this.konva.stage.position(newPos);

    // Update handle scaling to maintain constant screen size
    this.updateHandleScale();
    this.renderBg();
    this.callbacks.onZoomChange?.(newScale);
  };

  onKeyUp = (e: KeyboardEvent) => {
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

  // Pan with Space + drag or middle mouse button
  onPointerDown = (e: KonvaEventObject<PointerEvent>) => {
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

  onPointerMove = (_: KonvaEventObject<PointerEvent>) => {
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

    this.renderBg();

    this.lastPointerPosition = pointer;
  };

  onPointerUp = (_: KonvaEventObject<PointerEvent>) => {
    if (!this.konva?.stage) {
      return;
    }
    if (this.isPanning) {
      this.isPanning = false;
      this.konva.stage.container().style.cursor = this.isSpacePressed ? 'grab' : 'default';
    }
  };

  // Prevent context menu on right click
  onContextMenu = (e: MouseEvent) => e.preventDefault();

  // Image Management
  loadImage = (src: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      const img = new Image();

      img.crossOrigin = $crossOrigin.get();

      img.onload = () => {
        this.originalImage = img;
        this.displayImage();
        this.callbacks.onImageLoad?.();
        resolve();
      };

      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };

      img.src = src;
    });
  };

  private displayImage = () => {
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
  };

  private resizeCropBox = (handleName: HandleName, handleRect: Konva.Rect) => {
    if (!this.konva) {
      return;
    }

    let { newX, newY, newWidth, newHeight } = this.cropConstraints.aspectRatio
      ? this._resizeCropBoxWithAspectRatio(handleName, handleRect)
      : this._resizeCropBoxFree(handleName, handleRect);

    // Apply general constraints
    if (this.cropConstraints.maxWidth) {
      newWidth = Math.min(newWidth, this.cropConstraints.maxWidth);
    }
    if (this.cropConstraints.maxHeight) {
      newHeight = Math.min(newHeight, this.cropConstraints.maxHeight);
    }

    this.updateCropBox({
      x: newX,
      y: newY,
      width: newWidth,
      height: newHeight,
    });
  };

  private _resizeCropBoxFree = (handleName: HandleName, handleRect: Konva.Rect) => {
    if (!this.konva?.image.image) {
      throw new Error('Crop box or image not found');
    }
    const rect = this.konva.crop.overlay.clear;
    const imgWidth = this.konva.image.image.width();
    const imgHeight = this.konva.image.image.height();

    let newX = rect.x();
    let newY = rect.y();
    let newWidth = rect.width();
    let newHeight = rect.height();

    const handleX = handleRect.x() + handleRect.width() / 2;
    const handleY = handleRect.y() + handleRect.height() / 2;

    const minWidth = this.cropConstraints.minWidth ?? this.MIN_CROP_DIMENSION;
    const minHeight = this.cropConstraints.minHeight ?? this.MIN_CROP_DIMENSION;

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

  private _resizeCropBoxWithAspectRatio = (handleName: HandleName, handleRect: Konva.Rect) => {
    if (!this.konva?.image.image || !this.cropConstraints.aspectRatio || !this.cropBox) {
      throw new Error('Crop box, image, or aspect ratio not found');
    }
    const imgWidth = this.konva.image.image.width();
    const imgHeight = this.konva.image.image.height();
    const ratio = this.cropConstraints.aspectRatio;

    const handleX = handleRect.x() + handleRect.width() / 2;
    const handleY = handleRect.y() + handleRect.height() / 2;

    const minWidth = this.cropConstraints.minWidth ?? this.MIN_CROP_DIMENSION;
    const minHeight = this.cropConstraints.minHeight ?? this.MIN_CROP_DIMENSION;

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
    } = this._resizeCropBoxFree(handleName, handleRect);

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

  private positionHandle = (handleName: HandleName, handleRect: Konva.Rect) => {
    if (!this.konva || !this.cropBox) {
      return;
    }

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
  };

  private updateHandlePositions = () => {
    if (!this.konva) {
      return;
    }

    for (const [handleName, handleRect] of objectEntries(this.konva.crop.interaction.handles)) {
      this.positionHandle(handleName, handleRect);
    }
  };

  private updateCropOverlay = () => {
    if (!this.konva?.image.image || !this.cropBox) {
      return;
    }

    this.konva.crop.overlay.full.setAttrs({
      ...this.konva.image.image.getPosition(),
      ...this.konva.image.image.getSize(),
    });

    this.konva.crop.overlay.clear.setAttrs({ ...this.cropBox });
  };

  private updateHandleScale = () => {
    if (!this.konva) {
      return;
    }

    const scale = this.konva.stage.scaleX();
    const handleSize = this.CROP_HANDLE_SIZE / scale;
    const strokeWidth = this.CROP_HANDLE_STROKE_WIDTH / scale;

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

  // Crop Mode
  startCrop = (crop?: CropBox) => {
    if (!this.konva?.image.image || this.isCropping) {
      return;
    }

    // Calculate initial crop dimensions
    let cropX: number;
    let cropY: number;
    let cropWidth: number;
    let cropHeight: number;

    if (crop) {
      cropX = crop.x;
      cropY = crop.y;
      cropWidth = crop.width;
      cropHeight = crop.height;
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
      cropWidth = imgWidth * this.DEFAULT_CROP_BOX_SCALE;
      cropHeight = imgHeight * this.DEFAULT_CROP_BOX_SCALE;
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

    this.callbacks.onCropStart?.();
  };

  cancelCrop = () => {
    if (!this.isCropping || !this.konva) {
      return;
    }
    this.isCropping = false;
    this.konva.crop.interaction.group.visible(false);
    this.callbacks.onCropCancel?.();
  };

  applyCrop = () => {
    if (!this.isCropping || !this.cropBox || !this.konva) {
      return;
    }

    this.isCropping = false;
    this.konva.crop.interaction.group.visible(false);
    this.callbacks.onCropApply?.(this.cropBox);
  };

  resetCrop = () => {
    if (this.konva?.image.image) {
      this.updateCropBox({
        x: 0,
        y: 0,
        ...this.konva.image.image.size(),
      });
    }
    this.callbacks.onCropReset?.();
  };

  // Export
  exportImage = <T extends 'canvas' | 'blob' | 'dataURL'>(
    format: T = 'blob' as T
  ): Promise<
    T extends 'canvas' ? HTMLCanvasElement : T extends 'blob' ? Blob : T extends 'dataURL' ? string : never
  > => {
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

  // View Control
  setZoom = (scale: number, point?: { x: number; y: number }) => {
    if (!this.konva) {
      return;
    }

    scale = Math.max(this.ZOOM_MIN, Math.min(this.ZOOM_MAX, scale));

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
    this.updateHandleScale();

    this.renderBg();

    this.callbacks.onZoomChange?.(scale);
  };

  getZoom = (): number => {
    return this.konva?.stage.scaleX() || 1;
  };

  zoomIn = (point?: { x: number; y: number }) => {
    const currentZoom = this.getZoom();
    this.setZoom(currentZoom * this.ZOOM_BUTTON_FACTOR, point);
  };

  zoomOut = (point?: { x: number; y: number }) => {
    const currentZoom = this.getZoom();
    this.setZoom(currentZoom / this.ZOOM_BUTTON_FACTOR, point);
  };

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
    this.updateHandleScale();

    this.renderBg();

    this.callbacks.onZoomChange?.(1);
  };

  fitToContainer = () => {
    if (!this.konva?.image?.image) {
      return;
    }

    const containerWidth = this.konva.stage.width();
    const containerHeight = this.konva.stage.height();
    const imageWidth = this.konva.image.image.width();
    const imageHeight = this.konva.image.image.height();

    const scale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight) * this.FIT_TO_CONTAINER_PADDING;

    this.konva.stage.scale({ x: scale, y: scale });

    // Center the image
    const scaledWidth = imageWidth * scale;
    const scaledHeight = imageHeight * scale;

    this.konva.stage.position({
      x: (containerWidth - scaledWidth) / 2,
      y: (containerHeight - scaledHeight) / 2,
    });

    // Update handle scaling
    this.updateHandleScale();

    this.renderBg();

    this.callbacks.onZoomChange?.(scale);
  };

  // Configuration
  setCallbacks = (callbacks: EditorCallbacks, replace = false) => {
    if (replace) {
      this.callbacks = callbacks;
    } else {
      this.callbacks = { ...this.callbacks, ...callbacks };
    }
  };

  setCropAspectRatio = (ratio: number | undefined) => {
    // Update the constraint
    this.cropConstraints.aspectRatio = ratio;

    if (!this.konva?.image.image || !this.cropBox) {
      return;
    }

    const currentWidth = this.cropBox.width;
    const currentHeight = this.cropBox.height;
    const currentArea = currentWidth * currentHeight;

    if (ratio === undefined) {
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
    const minWidth = this.cropConstraints.minWidth ?? this.MIN_CROP_DIMENSION;
    const minHeight = this.cropConstraints.minHeight ?? this.MIN_CROP_DIMENSION;

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

  getCropAspectRatio = (): number | undefined => {
    return this.cropConstraints.aspectRatio;
  };

  // Utility
  resize = (width: number, height: number) => {
    if (!this.konva) {
      return;
    }

    this.konva.stage.width(width);
    this.konva.stage.height(height);

    this.renderBg();
  };

  destroy = () => {
    for (const unsubscribe of this.subscriptions) {
      unsubscribe();
    }

    // Cancel any ongoing crop operation
    if (this.isCropping) {
      this.cancelCrop();
    }

    // Remove all Konva event listeners by destroying the stage
    // This automatically removes all Konva event handlers
    this.konva?.stage.destroy();

    // Clear all references
    this.konva = null;
    this.originalImage = null;
    this.cropBox = null;
    this.callbacks = {};
  };
}
