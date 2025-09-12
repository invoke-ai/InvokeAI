import { TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL } from 'features/controlLayers/konva/patterns/transparency-checkerboard-pattern';
import Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';

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

type KonvaObjects = {
  stage: Konva.Stage;
  bg: {
    layer: Konva.Layer;
    patternRect: Konva.Rect;
  };
  image?: {
    layer: Konva.Layer;
    node: Konva.Image;
  };
  crop?: {
    layer: Konva.Layer;
    rect: Konva.Rect;
    overlay: Konva.Group;
    handles: Konva.Group;
    guides: Konva.Group;
  };
  frozenCrop?: {
    layer: Konva.Layer;
    overlay: Konva.Group;
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
  private appliedCrop: CropBox | null = null;
  private currentImageBlobUrl: string | null = null;

  // Constants
  private readonly MIN_CROP_DIMENSION = 64;
  private readonly ZOOM_WHEEL_FACTOR = 1.1;
  private readonly ZOOM_BUTTON_FACTOR = 1.2;
  private readonly CROP_HANDLE_SIZE = 8;
  private readonly CROP_HANDLE_STROKE_WIDTH = 1;
  private readonly FIT_TO_CONTAINER_PADDING = 0.9;
  private readonly DEFAULT_CROP_BOX_SCALE = 0.8;
  private readonly CORNER_HANDLE_NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
  private readonly EDGE_HANDLE_NAMES = ['top', 'right', 'bottom', 'left'];

  // Configuration
  private readonly ZOOM_MIN = 0.1;
  private readonly ZOOM_MAX = 10;

  private cropConstraints: CropConstraints = {
    minWidth: this.MIN_CROP_DIMENSION,
    minHeight: this.MIN_CROP_DIMENSION,
  };
  private callbacks: EditorCallbacks = {};

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

    const bgLayer = new Konva.Layer();
    const bgPatternRect = new Konva.Rect();
    bgLayer.add(bgPatternRect);
    const bgImage = new Image();
    bgImage.onload = () => {
      bgPatternRect.fillPatternImage(bgImage);
      this.renderBg();
    };
    bgImage.src = TRANSPARENCY_CHECKERBOARD_PATTERN_DARK_DATAURL;

    stage.add(bgLayer);

    this.konva = {
      stage,
      bg: {
        layer: bgLayer,
        patternRect: bgPatternRect,
      },
    };
    // Setup mouse event handlers
    this.setupStageEvents();
  };

  renderBg = () => {
    if (!this.konva) {
      return;
    }
    const scale = this.konva.stage.scaleX();
    const patternScale = 1 / scale;
    const { x, y } = this.konva.stage.getPosition();
    const { width, height } = this.konva.stage.size();

    this.konva.bg.patternRect.setAttrs({
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
        if (this.konva.crop.rect.isDragging()) {
          this.konva.crop.rect.stopDrag();
        }
        this.konva.crop.handles.children.forEach((handle) => {
          if (handle.isDragging()) {
            handle.stopDrag();
          }
        });
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
  loadImage = (src: string | File | Blob): Promise<void> => {
    return new Promise((resolve, reject) => {
      // Clean up previous blob URL if it exists
      if (this.currentImageBlobUrl) {
        URL.revokeObjectURL(this.currentImageBlobUrl);
        this.currentImageBlobUrl = null;
      }

      const img = new Image();

      // Set crossOrigin to avoid CORS issues when exporting
      if (typeof src === 'string') {
        img.crossOrigin = 'anonymous';
      }

      img.onload = () => {
        this.originalImage = img;
        this.displayImage();
        this.callbacks.onImageLoad?.();
        resolve();
      };

      img.onerror = () => {
        // Clean up blob URL on error
        if (this.currentImageBlobUrl) {
          URL.revokeObjectURL(this.currentImageBlobUrl);
          this.currentImageBlobUrl = null;
        }
        reject(new Error('Failed to load image'));
      };

      if (typeof src === 'string') {
        img.src = src;
      } else if (src instanceof File || src instanceof Blob) {
        const url = URL.createObjectURL(src);
        this.currentImageBlobUrl = url;
        img.src = url;
      }
    });
  };

  private displayImage = () => {
    if (!this.originalImage || !this.konva) {
      return;
    }

    // Clear existing image
    if (this.konva.image) {
      this.konva.image.node.destroy();
      this.konva.image.layer.destroy();
      this.konva.image = undefined;
    }

    // Create image layer and node - always show full image
    const imageLayer = new Konva.Layer();
    const imageNode = new Konva.Image({
      image: this.originalImage,
      x: 0,
      y: 0,
      width: this.originalImage.width,
      height: this.originalImage.height,
    });

    imageLayer.add(imageNode);
    this.konva.stage.add(imageLayer);

    // Store references
    this.konva.image = {
      layer: imageLayer,
      node: imageNode,
    };

    imageLayer.batchDraw();

    // If there's an applied crop, create frozen overlay
    if (this.appliedCrop) {
      this.createFrozenCropOverlay();
    }

    // Center image at 100% zoom
    this.resetView();
  };

  // Crop Mode
  startCrop = (crop?: CropBox) => {
    if (!this.konva?.image || this.isCropping) {
      return;
    }

    // Remove frozen crop overlay if it exists
    if (this.konva.frozenCrop) {
      this.konva.frozenCrop.layer.destroy();
      this.konva.frozenCrop = undefined;
    }

    this.isCropping = true;

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
    } else if (this.appliedCrop) {
      // Use the applied crop as starting point
      cropX = this.appliedCrop.x;
      cropY = this.appliedCrop.y;
      cropWidth = this.appliedCrop.width;
      cropHeight = this.appliedCrop.height;
    } else {
      // Create default crop box (centered, 80% of image)
      const imgWidth = this.konva.image.node.width();
      const imgHeight = this.konva.image.node.height();
      cropWidth = imgWidth * this.DEFAULT_CROP_BOX_SCALE;
      cropHeight = imgHeight * this.DEFAULT_CROP_BOX_SCALE;
      cropX = (imgWidth - cropWidth) / 2;
      cropY = (imgHeight - cropHeight) / 2;
    }

    this.createCropBox(cropX, cropY, cropWidth, cropHeight);

    this.callbacks.onCropStart?.();
    this.callbacks.onCropBoxChange?.({
      x: cropX,
      y: cropY,
      width: cropWidth,
      height: cropHeight,
    });
  };

  private createCropBox = (x: number, y: number, width: number, height: number) => {
    if (!this.konva?.image) {
      return;
    }

    // Clear existing crop if any
    if (this.konva.crop) {
      this.konva.crop.layer.destroy();
      this.konva.crop = undefined;
    }

    const imgWidth = this.konva.image.node.width();
    const imgHeight = this.konva.image.node.height();

    // Create crop layer
    const cropLayer = new Konva.Layer();

    // Create overlay group with composite operation
    const overlay = new Konva.Group();

    // Create full overlay
    const fullOverlay = new Konva.Rect({
      x: 0,
      y: 0,
      width: imgWidth,
      height: imgHeight,
      fill: 'black',
      opacity: 0.7,
    });

    // Create clear rectangle for crop area using composite operation
    const clearRect = new Konva.Rect({
      x: x,
      y: y,
      width: width,
      height: height,
      fill: 'black',
      globalCompositeOperation: 'destination-out',
    });

    overlay.add(fullOverlay);
    overlay.add(clearRect);

    // Create crop rectangle
    const rect = new Konva.Rect({
      x: x,
      y: y,
      width: width,
      height: height,
      stroke: 'white',
      strokeWidth: this.CROP_HANDLE_STROKE_WIDTH,
      strokeScaleEnabled: false,
      draggable: true,
    });

    // Create handles group
    const handles = new Konva.Group();

    // Create guides group
    const guides = new Konva.Group();

    // Store all crop objects together
    this.konva.crop = {
      layer: cropLayer,
      rect: rect,
      overlay: overlay,
      handles: handles,
      guides: guides,
    };

    // Create handles and guides
    this.createCropHandles();
    this.createCropGuides();

    // Setup crop box events
    this.setupCropBoxEvents();

    // Add to layer
    cropLayer.add(overlay);
    cropLayer.add(rect);
    cropLayer.add(guides);
    cropLayer.add(handles);

    // Add layer to stage
    this.konva.stage.add(cropLayer);

    // Apply current scale to handles
    this.updateHandleScale();

    cropLayer.batchDraw();
  };

  private createCropGuides = () => {
    if (!this.konva?.crop) {
      return;
    }

    const rect = this.konva.crop.rect;
    const guides = this.konva.crop.guides;

    const x = rect.x();
    const y = rect.y();
    const width = rect.width();
    const height = rect.height();

    const guideConfig = {
      stroke: 'rgba(255, 255, 255, 0.5)',
      strokeWidth: this.CROP_HANDLE_STROKE_WIDTH,
      strokeScaleEnabled: false,
      listening: false,
    };

    // Vertical lines (thirds)
    const verticalThird = width / 3;
    guides.add(
      new Konva.Line({
        points: [x + verticalThird, y, x + verticalThird, y + height],
        ...guideConfig,
      })
    );
    guides.add(
      new Konva.Line({
        points: [x + verticalThird * 2, y, x + verticalThird * 2, y + height],
        ...guideConfig,
      })
    );

    // Horizontal lines (thirds)
    const horizontalThird = height / 3;
    guides.add(
      new Konva.Line({
        points: [x, y + horizontalThird, x + width, y + horizontalThird],
        ...guideConfig,
      })
    );
    guides.add(
      new Konva.Line({
        points: [x, y + horizontalThird * 2, x + width, y + horizontalThird * 2],
        ...guideConfig,
      })
    );
  };

  private createCropHandles = () => {
    if (!this.konva?.crop) {
      return;
    }

    const rect = this.konva.crop.rect;
    const handles = this.konva.crop.handles;
    const scale = this.konva.stage.scaleX();
    const handleSize = this.CROP_HANDLE_SIZE / scale;
    const handleConfig = {
      width: handleSize,
      height: handleSize,
      fill: 'white',
      stroke: 'black',
      strokeWidth: this.CROP_HANDLE_STROKE_WIDTH,
      strokeScaleEnabled: true,
    };

    // Corner handles
    const corners = [
      { name: 'top-left', x: 0, y: 0 },
      { name: 'top-right', x: 1, y: 0 },
      { name: 'bottom-right', x: 1, y: 1 },
      { name: 'bottom-left', x: 0, y: 1 },
    ];

    corners.forEach((corner) => {
      const handle = new Konva.Rect({
        ...handleConfig,
        name: corner.name,
        x: rect.x() + corner.x * rect.width() - handleSize / 2,
        y: rect.y() + corner.y * rect.height() - handleSize / 2,
        draggable: true,
      });

      this.setupHandleEvents(handle);
      handles.add(handle);
    });

    // Edge handles
    const edges = [
      { name: 'top', x: 0.5, y: 0 },
      { name: 'right', x: 1, y: 0.5 },
      { name: 'bottom', x: 0.5, y: 1 },
      { name: 'left', x: 0, y: 0.5 },
    ];

    edges.forEach((edge) => {
      const handle = new Konva.Rect({
        ...handleConfig,
        name: edge.name,
        x: rect.x() + edge.x * rect.width() - handleSize / 2,
        y: rect.y() + edge.y * rect.height() - handleSize / 2,
        draggable: true,
      });

      this.setupHandleEvents(handle);
      handles.add(handle);
    });
  };

  private setupCropBoxEvents = () => {
    if (!this.konva?.crop) {
      return;
    }
    const stage = this.konva.stage;
    const rect = this.konva.crop.rect;
    const image = this.konva.image;
    if (!image) {
      return;
    }

    // Prevent crop box dragging when panning
    rect.on('dragstart', (e) => {
      if (this.isSpacePressed || this.isPanning) {
        e.target.stopDrag();
        return false;
      }
    });

    // Crop box dragging
    rect.on('dragmove', () => {
      const imgWidth = image.node.width();
      const imgHeight = image.node.height();

      // Constrain to image bounds
      const x = Math.max(0, Math.min(rect.x(), imgWidth - rect.width()));
      const y = Math.max(0, Math.min(rect.y(), imgHeight - rect.height()));

      rect.x(x);
      rect.y(y);

      this.updateCropOverlay();
      this.updateHandlePositions();
      this.updateCropGuides();

      this.callbacks.onCropBoxChange?.({
        x,
        y,
        width: rect.width(),
        height: rect.height(),
      });
    });

    // Cursor styles
    rect.on('mouseenter', () => {
      if (!this.isSpacePressed) {
        stage.container().style.cursor = 'move';
      }
    });

    rect.on('mouseleave', () => {
      if (!this.isSpacePressed) {
        stage.container().style.cursor = 'default';
      }
    });
  };

  private setupHandleEvents = (handle: Konva.Rect) => {
    if (!this.konva) {
      return;
    }
    const stage = this.konva.stage;
    const handleName = handle.name();

    // Prevent handle dragging when panning
    handle.on('dragstart', (e) => {
      if (this.isSpacePressed || this.isPanning) {
        e.target.stopDrag();
        return false;
      }
    });

    // Set cursor based on handle type
    handle.on('mouseenter', () => {
      if (!this.isSpacePressed) {
        let cursor = 'pointer';
        if (handleName.includes('top-left') || handleName.includes('bottom-right')) {
          cursor = 'nwse-resize';
        } else if (handleName.includes('top-right') || handleName.includes('bottom-left')) {
          cursor = 'nesw-resize';
        } else if (handleName.includes('top') || handleName.includes('bottom')) {
          cursor = 'ns-resize';
        } else if (handleName.includes('left') || handleName.includes('right')) {
          cursor = 'ew-resize';
        }
        stage.container().style.cursor = cursor;
      }
    });

    handle.on('mouseleave', () => {
      if (!this.isSpacePressed) {
        stage.container().style.cursor = 'default';
      }
    });

    // Handle dragging
    handle.on('dragmove', () => {
      this.resizeCropBox(handle);
    });
  };

  private resizeCropBox = (handle: Konva.Rect) => {
    if (!this.konva?.crop || !this.konva?.image) {
      return;
    }

    const rect = this.konva.crop.rect;

    let { newX, newY, newWidth, newHeight } = this.cropConstraints.aspectRatio
      ? this._resizeCropBoxWithAspectRatio(handle)
      : this._resizeCropBoxFree(handle);

    // Apply general constraints
    if (this.cropConstraints.maxWidth) {
      newWidth = Math.min(newWidth, this.cropConstraints.maxWidth);
    }
    if (this.cropConstraints.maxHeight) {
      newHeight = Math.min(newHeight, this.cropConstraints.maxHeight);
    }

    // Update crop rect
    rect.x(newX);
    rect.y(newY);
    rect.width(newWidth);
    rect.height(newHeight);

    // Update overlay, handles, and guides
    this.updateCropOverlay();
    this.updateHandlePositions();
    this.updateCropGuides();

    // Reset handle position to follow crop box
    this.positionHandle(handle);

    this.callbacks.onCropBoxChange?.({
      x: newX,
      y: newY,
      width: newWidth,
      height: newHeight,
    });
  };

  private _resizeCropBoxFree = (handle: Konva.Rect) => {
    if (!this.konva?.crop || !this.konva?.image) {
      throw new Error('Crop box or image not found');
    }
    const rect = this.konva.crop.rect;
    const handleName = handle.name();
    const imgWidth = this.konva.image.node.width();
    const imgHeight = this.konva.image.node.height();

    let newX = rect.x();
    let newY = rect.y();
    let newWidth = rect.width();
    let newHeight = rect.height();

    const handleX = handle.x() + handle.width() / 2;
    const handleY = handle.y() + handle.height() / 2;

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

  private _resizeCropBoxWithAspectRatio = (handle: Konva.Rect) => {
    if (!this.konva?.crop || !this.konva?.image || !this.cropConstraints.aspectRatio) {
      throw new Error('Crop box, image, or aspect ratio not found');
    }
    const rect = this.konva.crop.rect;
    const handleName = handle.name();
    const imgWidth = this.konva.image.node.width();
    const imgHeight = this.konva.image.node.height();
    const ratio = this.cropConstraints.aspectRatio;

    const handleX = handle.x() + handle.width() / 2;
    const handleY = handle.y() + handle.height() / 2;

    const minWidth = this.cropConstraints.minWidth ?? this.MIN_CROP_DIMENSION;
    const minHeight = this.cropConstraints.minHeight ?? this.MIN_CROP_DIMENSION;

    // Early boundary check for aspect ratio mode
    const atLeftEdge = rect.x() <= 0;
    const atRightEdge = rect.x() + rect.width() >= imgWidth;
    const atTopEdge = rect.y() <= 0;
    const atBottomEdge = rect.y() + rect.height() >= imgHeight;

    if (
      (handleName === 'left' && atLeftEdge && handleX >= rect.x()) ||
      (handleName === 'right' && atRightEdge && handleX <= rect.x() + rect.width()) ||
      (handleName === 'top' && atTopEdge && handleY >= rect.y()) ||
      (handleName === 'bottom' && atBottomEdge && handleY <= rect.y() + rect.height())
    ) {
      return { newX: rect.x(), newY: rect.y(), newWidth: rect.width(), newHeight: rect.height() };
    }

    const { newX: freeX, newY: freeY, newWidth: freeWidth, newHeight: freeHeight } = this._resizeCropBoxFree(handle);
    let newX = freeX;
    let newY = freeY;
    let newWidth = freeWidth;
    let newHeight = freeHeight;

    const oldX = rect.x();
    const oldY = rect.y();
    const oldWidth = rect.width();
    const oldHeight = rect.height();

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

    const isCornerHandle = this.CORNER_HANDLE_NAMES.includes(handleName);

    // Calculate new dimensions maintaining aspect ratio
    if (this.EDGE_HANDLE_NAMES.includes(handleName) && !isCornerHandle) {
      if (handleName === 'left' || handleName === 'right') {
        newHeight = newWidth / ratio;
        newY = anchorY - newHeight / 2;
      } else {
        // top or bottom
        newWidth = newHeight * ratio;
        newX = anchorX - newWidth / 2;
      }
    } else if (isCornerHandle) {
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

  private positionHandle = (handle: Konva.Rect) => {
    if (!this.konva?.crop) {
      return;
    }

    const rect = this.konva.crop.rect;
    const handleName = handle.name();
    const handleSize = handle.width();

    let x = rect.x();
    let y = rect.y();

    if (handleName.includes('right')) {
      x += rect.width();
    } else if (!handleName.includes('left')) {
      x += rect.width() / 2;
    }

    if (handleName.includes('bottom')) {
      y += rect.height();
    } else if (!handleName.includes('top')) {
      y += rect.height() / 2;
    }

    handle.x(x - handleSize / 2);
    handle.y(y - handleSize / 2);
  };

  private updateHandlePositions = () => {
    if (!this.konva?.crop) {
      return;
    }

    this.konva.crop.handles.children.forEach((handle) => {
      if (handle instanceof Konva.Rect) {
        this.positionHandle(handle);
      }
    });
  };

  private updateCropGuides = () => {
    if (!this.konva?.crop) {
      return;
    }

    const rect = this.konva.crop.rect;
    const x = rect.x();
    const y = rect.y();
    const width = rect.width();
    const height = rect.height();

    const lines = this.konva.crop.guides.children;
    if (lines.length < 4) {
      return;
    }

    // Update vertical lines
    const verticalThird = width / 3;
    const line0 = lines[0];
    const line1 = lines[1];
    if (line0 instanceof Konva.Line) {
      line0.points([x + verticalThird, y, x + verticalThird, y + height]);
    }
    if (line1 instanceof Konva.Line) {
      line1.points([x + verticalThird * 2, y, x + verticalThird * 2, y + height]);
    }

    // Update horizontal lines
    const horizontalThird = height / 3;
    const line2 = lines[2];
    const line3 = lines[3];
    if (line2 instanceof Konva.Line) {
      line2.points([x, y + horizontalThird, x + width, y + horizontalThird]);
    }
    if (line3 instanceof Konva.Line) {
      line3.points([x, y + horizontalThird * 2, x + width, y + horizontalThird * 2]);
    }
  };

  private updateCropOverlay = () => {
    if (!this.konva?.crop) {
      return;
    }

    const rect = this.konva.crop.rect;
    const x = rect.x();
    const y = rect.y();
    const width = rect.width();
    const height = rect.height();

    const nodes = this.konva.crop.overlay.children;

    // Update clear rectangle position (the cutout)
    if (nodes.length > 1) {
      const clearRect = nodes[1];
      if (clearRect instanceof Konva.Rect) {
        clearRect.x(x);
        clearRect.y(y);
        clearRect.width(width);
        clearRect.height(height);
      }
    }

    this.konva.crop.layer.batchDraw();
  };

  private updateHandleScale = () => {
    if (!this.konva?.crop) {
      return;
    }

    const scale = this.konva.stage.scaleX();
    const handleSize = this.CROP_HANDLE_SIZE / scale;
    const strokeWidth = this.CROP_HANDLE_STROKE_WIDTH / scale;

    // Update each handle's size and stroke to maintain constant screen size
    this.konva.crop.handles.children.forEach((handle) => {
      if (handle instanceof Konva.Rect) {
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
    });

    this.konva.crop.layer.batchDraw();
  };

  private freezeCropOverlay = () => {
    if (!this.konva?.crop || !this.konva?.image) {
      return;
    }

    const imgWidth = this.konva.image.node.width();
    const imgHeight = this.konva.image.node.height();
    const cropX = this.konva.crop.rect.x();
    const cropY = this.konva.crop.rect.y();
    const cropWidth = this.konva.crop.rect.width();
    const cropHeight = this.konva.crop.rect.height();

    // Create a new frozen overlay layer
    const frozenLayer = new Konva.Layer();
    const frozenOverlay = new Konva.Group();

    // Create full overlay
    const fullOverlay = new Konva.Rect({
      x: 0,
      y: 0,
      width: imgWidth,
      height: imgHeight,
      fill: 'black',
      opacity: 0.7,
    });

    // Create clear rectangle for crop area
    const clearRect = new Konva.Rect({
      x: cropX,
      y: cropY,
      width: cropWidth,
      height: cropHeight,
      fill: 'black',
      globalCompositeOperation: 'destination-out',
    });

    frozenOverlay.add(fullOverlay);
    frozenOverlay.add(clearRect);
    frozenLayer.add(frozenOverlay);

    // Add frozen layer to stage
    this.konva.stage.add(frozenLayer);

    // Store reference to frozen overlay
    this.konva.frozenCrop = {
      layer: frozenLayer,
      overlay: frozenOverlay,
    };

    // Remove the interactive crop layer
    this.resetEphemeralCropState();

    frozenLayer.batchDraw();
  };

  private createFrozenCropOverlay = () => {
    if (!this.appliedCrop || !this.konva?.image) {
      return;
    }

    const imgWidth = this.konva.image.node.width();
    const imgHeight = this.konva.image.node.height();

    // Create a frozen overlay layer
    const frozenLayer = new Konva.Layer();
    const frozenOverlay = new Konva.Group();

    // Create full overlay
    const fullOverlay = new Konva.Rect({
      x: 0,
      y: 0,
      width: imgWidth,
      height: imgHeight,
      fill: 'black',
      opacity: 0.7,
    });

    // Create clear rectangle for crop area
    const clearRect = new Konva.Rect({
      x: this.appliedCrop.x,
      y: this.appliedCrop.y,
      width: this.appliedCrop.width,
      height: this.appliedCrop.height,
      fill: 'black',
      globalCompositeOperation: 'destination-out',
    });

    frozenOverlay.add(fullOverlay);
    frozenOverlay.add(clearRect);
    frozenLayer.add(frozenOverlay);

    // Add frozen layer to stage
    this.konva.stage.add(frozenLayer);

    // Store reference to frozen overlay
    this.konva.frozenCrop = {
      layer: frozenLayer,
      overlay: frozenOverlay,
    };

    frozenLayer.batchDraw();
  };

  resetEphemeralCropState = () => {
    this.isCropping = false;
    if (this.konva?.crop) {
      this.konva.crop.layer.destroy();
      this.konva.crop = undefined;
    }
  };

  cancelCrop = () => {
    if (!this.isCropping || !this.konva?.crop) {
      return;
    }
    this.resetEphemeralCropState();
    this.callbacks.onCropCancel?.();
  };

  applyCrop = () => {
    if (!this.isCropping || !this.konva?.crop) {
      return;
    }

    const rect = this.konva.crop.rect;

    // Store the crop dimensions
    this.appliedCrop = {
      x: rect.x(),
      y: rect.y(),
      width: rect.width(),
      height: rect.height(),
    };

    // Freeze the crop overlay instead of redisplaying image
    this.freezeCropOverlay();

    this.isCropping = false;
    this.callbacks.onCropApply?.(this.appliedCrop);
  };

  resetCrop = () => {
    this.appliedCrop = null;

    // Remove frozen crop overlay if it exists
    if (this.konva?.frozenCrop) {
      this.konva.frozenCrop.layer.destroy();
      this.konva.frozenCrop = undefined;
    }

    this.callbacks.onCropReset?.();
  };

  hasCrop = (): boolean => {
    return !!this.appliedCrop;
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
        if (this.appliedCrop) {
          canvas.width = this.appliedCrop.width;
          canvas.height = this.appliedCrop.height;

          ctx.drawImage(
            this.originalImage,
            this.appliedCrop.x,
            this.appliedCrop.y,
            this.appliedCrop.width,
            this.appliedCrop.height,
            0,
            0,
            this.appliedCrop.width,
            this.appliedCrop.height
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
    if (!this.konva?.image) {
      return;
    }

    this.konva.stage.scale({ x: 1, y: 1 });

    // Center the image
    const containerWidth = this.konva.stage.width();
    const containerHeight = this.konva.stage.height();
    const imageWidth = this.konva.image.node.width();
    const imageHeight = this.konva.image.node.height();

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
    if (!this.konva?.image) {
      return;
    }

    const containerWidth = this.konva.stage.width();
    const containerHeight = this.konva.stage.height();
    const imageWidth = this.konva.image.node.width();
    const imageHeight = this.konva.image.node.height();

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

    // If we're currently cropping, adjust the crop box to match the new ratio
    if (this.isCropping && this.konva?.crop && this.konva?.image) {
      const rect = this.konva.crop.rect;
      const currentWidth = rect.width();
      const currentHeight = rect.height();
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
      const imgWidth = this.konva.image.node.width();
      const imgHeight = this.konva.image.node.height();

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
      const currentCenterX = rect.x() + currentWidth / 2;
      const currentCenterY = rect.y() + currentHeight / 2;

      let newX = currentCenterX - newWidth / 2;
      let newY = currentCenterY - newHeight / 2;

      // Ensure the crop box stays within image bounds
      newX = Math.max(0, Math.min(newX, imgWidth - newWidth));
      newY = Math.max(0, Math.min(newY, imgHeight - newHeight));

      // Update the crop box
      rect.x(newX);
      rect.y(newY);
      rect.width(newWidth);
      rect.height(newHeight);

      // Update all visual elements
      this.updateCropOverlay();
      this.updateHandlePositions();
      this.updateCropGuides();

      // Notify callback
      this.callbacks.onCropBoxChange?.({
        x: newX,
        y: newY,
        width: newWidth,
        height: newHeight,
      });

      // Force a redraw
      this.konva.crop.layer.batchDraw();
    }
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

    // Clean up blob URL if it exists
    if (this.currentImageBlobUrl) {
      URL.revokeObjectURL(this.currentImageBlobUrl);
      this.currentImageBlobUrl = null;
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
    this.appliedCrop = null;
    this.callbacks = {};
  };
}
