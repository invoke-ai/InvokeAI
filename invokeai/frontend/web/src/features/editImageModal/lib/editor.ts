import Konva from "konva";

interface CropConstraints {
    minWidth?: number;
    minHeight?: number;
    maxWidth?: number;
    maxHeight?: number;
    aspectRatio?: number;
}

interface EditorCallbacks {
    onCropChange?: (crop: {
        x: number;
        y: number;
        width: number;
        height: number;
    }) => void;
    onZoomChange?: (zoom: number) => void;
    onImageLoad?: () => void;
}

interface CropData {
    x: number;
    y: number;
    width: number;
    height: number;
}

interface KonvaObjects {
    stage: Konva.Stage;
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
}

export class Editor {
    private konva?: KonvaObjects;
    private originalImage?: HTMLImageElement;
    private isInCropMode = false;
    private appliedCrop?: CropData;

    // Configuration
    private zoomMin = 0.1;
    private zoomMax = 10;
    private cropConstraints: CropConstraints = {
        minWidth: 64,
        minHeight: 64,
    };
    private callbacks: EditorCallbacks = {};

    // State
    private isPanning = false;
    private lastPointerPosition?: { x: number; y: number };
    private isSpacePressed = false;
    private keydownHandler?: (e: KeyboardEvent) => void;
    private keyupHandler?: (e: KeyboardEvent) => void;
    private contextMenuHandler?: (e: Event) => void;
    private currentImageBlobUrl?: string;
    private wheelHandler?: (e: WheelEvent) => void;

    init = (container: HTMLDivElement) => {
        // Create stage
        this.konva = {
            stage: new Konva.Stage({
                container: container,
                width: container.clientWidth,
                height: container.clientHeight,
            }),
        };

        // Setup mouse event handlers
        this.setupStageEvents();
    };

    private setupStageEvents = () => {
        if (!this.konva) {
return;
}
        const stage = this.konva.stage;

        // Zoom with mouse wheel
        this.wheelHandler = (e: WheelEvent) => {
            e.preventDefault();

            const oldScale = stage.scaleX();
            const pointer = stage.getPointerPosition();

            if (!pointer) {
return;
}

            const mousePointTo = {
                x: (pointer.x - stage.x()) / oldScale,
                y: (pointer.y - stage.y()) / oldScale,
            };

            const direction = e.deltaY > 0 ? -1 : 1;
            const scaleBy = 1.1;
            let newScale =
                direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;

            // Apply zoom limits
            newScale = Math.max(this.zoomMin, Math.min(this.zoomMax, newScale));

            stage.scale({ x: newScale, y: newScale });

            const newPos = {
                x: pointer.x - mousePointTo.x * newScale,
                y: pointer.y - mousePointTo.y * newScale,
            };
            stage.position(newPos);

            // Update handle scaling to maintain constant screen size
            this.updateHandleScale();

            this.callbacks.onZoomChange?.(newScale);
        };

        stage
            .container()
            .addEventListener("wheel", this.wheelHandler, { passive: false });

        // Track Space key press
        this.keydownHandler = (e: KeyboardEvent) => {
            if (e.code === "Space" && !this.isSpacePressed) {
                e.preventDefault();
                this.isSpacePressed = true;
                if (stage) {
                    stage.container().style.cursor = "grab";
                }
            }
        };

        this.keyupHandler = (e: KeyboardEvent) => {
            if (e.code === "Space") {
                e.preventDefault();
                this.isSpacePressed = false;
                this.isPanning = false;
                if (stage) {
                    stage.container().style.cursor = "default";
                }
            }
        };

        window.addEventListener("keydown", this.keydownHandler);
        window.addEventListener("keyup", this.keyupHandler);

        // Pan with Space + drag or middle mouse button
        stage.on("mousedown", (e) => {
            if (this.isSpacePressed || e.evt.button === 1) {
                e.evt.preventDefault();
                e.evt.stopPropagation();
                this.isPanning = true;
                this.lastPointerPosition =
                    stage.getPointerPosition() || undefined;
                stage.container().style.cursor = "grabbing";

                // Stop any active drags on crop elements
                if (this.konva?.crop) {
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
        });

        stage.on("mousemove", () => {
            if (!this.isPanning || !this.lastPointerPosition) {
return;
}

            const pointer = stage.getPointerPosition();
            if (!pointer) {
return;
}

            const dx = pointer.x - this.lastPointerPosition.x;
            const dy = pointer.y - this.lastPointerPosition.y;

            stage.x(stage.x() + dx);
            stage.y(stage.y() + dy);

            this.lastPointerPosition = pointer;
        });

        stage.on("mouseup", () => {
            if (this.isPanning) {
                this.isPanning = false;
                stage.container().style.cursor = this.isSpacePressed
                    ? "grab"
                    : "default";
            }
        });

        // Prevent context menu on right click
        this.contextMenuHandler = (e: Event) => e.preventDefault();
        stage
            .container()
            .addEventListener("contextmenu", this.contextMenuHandler);
    };

    // Image Management
    loadImage = async (src: string | File | Blob): Promise<void> => {
        return new Promise((resolve, reject) => {
            // Clean up previous blob URL if it exists
            if (this.currentImageBlobUrl) {
                URL.revokeObjectURL(this.currentImageBlobUrl);
                this.currentImageBlobUrl = undefined;
            }

            const img = new Image();
            
            // Set crossOrigin to avoid CORS issues when exporting
            if (typeof src === "string") {
                img.crossOrigin = "anonymous";
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
                    this.currentImageBlobUrl = undefined;
                }
                reject(new Error("Failed to load image"));
            };

            if (typeof src === "string") {
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

        // Create image layer and node
        const imageLayer = new Konva.Layer();
        let imageNode: Konva.Image;

        if (this.appliedCrop) {
            imageNode = new Konva.Image({
                image: this.originalImage,
                x: 0,
                y: 0,
                width: this.appliedCrop.width,
                height: this.appliedCrop.height,
                crop: {
                    x: this.appliedCrop.x,
                    y: this.appliedCrop.y,
                    width: this.appliedCrop.width,
                    height: this.appliedCrop.height,
                },
            });
        } else {
            imageNode = new Konva.Image({
                image: this.originalImage,
                x: 0,
                y: 0,
                width: this.originalImage.width,
                height: this.originalImage.height,
            });
        }

        imageLayer.add(imageNode);
        this.konva.stage.add(imageLayer);

        // Store references
        this.konva.image = {
            layer: imageLayer,
            node: imageNode,
        };

        imageLayer.batchDraw();

        // Center image at 100% zoom
        this.resetView();
    };

    // Crop Mode
    startCrop = () => {
        if (!this.konva?.image || this.isInCropMode) {
return;
}

        this.isInCropMode = true;

        // Calculate initial crop dimensions
        let cropX: number; let cropY: number; let cropWidth: number; let cropHeight: number;

        if (this.appliedCrop) {
            // When cropped, start with full visible area
            cropX = 0;
            cropY = 0;
            cropWidth = this.appliedCrop.width;
            cropHeight = this.appliedCrop.height;
        } else {
            // Create default crop box (centered, 80% of image)
            const imgWidth = this.konva.image.node.width();
            const imgHeight = this.konva.image.node.height();
            cropWidth = imgWidth * 0.8;
            cropHeight = imgHeight * 0.8;
            cropX = (imgWidth - cropWidth) / 2;
            cropY = (imgHeight - cropHeight) / 2;
        }

        this.createCropBox(cropX, cropY, cropWidth, cropHeight);
    };

    private createCropBox = (
        x: number,
        y: number,
        width: number,
        height: number,
    ) => {
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
            fill: "black",
            opacity: 0.5,
        });

        // Create clear rectangle for crop area using composite operation
        const clearRect = new Konva.Rect({
            x: x,
            y: y,
            width: width,
            height: height,
            fill: "black",
            globalCompositeOperation: "destination-out",
        });

        overlay.add(fullOverlay);
        overlay.add(clearRect);

        // Create crop rectangle
        const rect = new Konva.Rect({
            x: x,
            y: y,
            width: width,
            height: height,
            stroke: "white",
            strokeWidth: 1,
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
            stroke: "rgba(255, 255, 255, 0.5)",
            strokeWidth: 1,
            strokeScaleEnabled: false,
            listening: false,
        };

        // Vertical lines (thirds)
        const verticalThird = width / 3;
        guides.add(
            new Konva.Line({
                points: [x + verticalThird, y, x + verticalThird, y + height],
                ...guideConfig,
            }),
        );
        guides.add(
            new Konva.Line({
                points: [
                    x + verticalThird * 2,
                    y,
                    x + verticalThird * 2,
                    y + height,
                ],
                ...guideConfig,
            }),
        );

        // Horizontal lines (thirds)
        const horizontalThird = height / 3;
        guides.add(
            new Konva.Line({
                points: [
                    x,
                    y + horizontalThird,
                    x + width,
                    y + horizontalThird,
                ],
                ...guideConfig,
            }),
        );
        guides.add(
            new Konva.Line({
                points: [
                    x,
                    y + horizontalThird * 2,
                    x + width,
                    y + horizontalThird * 2,
                ],
                ...guideConfig,
            }),
        );
    };

    private createCropHandles = () => {
        if (!this.konva?.crop) {
return;
}

        const rect = this.konva.crop.rect;
        const handles = this.konva.crop.handles;
        const scale = this.konva.stage.scaleX();
        const handleSize = 8 / scale;
        const handleConfig = {
            width: handleSize,
            height: handleSize,
            fill: "white",
            stroke: "black",
            strokeWidth: 1 / scale,
            strokeScaleEnabled: false,
        };

        // Corner handles
        const corners = [
            { name: "top-left", x: 0, y: 0 },
            { name: "top-right", x: 1, y: 0 },
            { name: "bottom-right", x: 1, y: 1 },
            { name: "bottom-left", x: 0, y: 1 },
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
            { name: "top", x: 0.5, y: 0 },
            { name: "right", x: 1, y: 0.5 },
            { name: "bottom", x: 0.5, y: 1 },
            { name: "left", x: 0, y: 0.5 },
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
        rect.on("dragstart", (e) => {
            if (this.isSpacePressed || this.isPanning) {
                e.target.stopDrag();
                return false;
            }
        });

        // Crop box dragging
        rect.on("dragmove", () => {
            const imgWidth = image.node.width();
            const imgHeight = image.node.height();

            // Constrain to image bounds
            const x = Math.max(0, Math.min(rect.x(), imgWidth - rect.width()));
            const y = Math.max(
                0,
                Math.min(rect.y(), imgHeight - rect.height()),
            );

            rect.x(x);
            rect.y(y);

            this.updateCropOverlay();
            this.updateHandlePositions();
            this.updateCropGuides();

            this.callbacks.onCropChange?.({
                x,
                y,
                width: rect.width(),
                height: rect.height(),
            });
        });

        // Cursor styles
        rect.on("mouseenter", () => {
            if (!this.isSpacePressed) {
                stage.container().style.cursor = "move";
            }
        });

        rect.on("mouseleave", () => {
            if (!this.isSpacePressed) {
                stage.container().style.cursor = "default";
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
        handle.on("dragstart", (e) => {
            if (this.isSpacePressed || this.isPanning) {
                e.target.stopDrag();
                return false;
            }
        });

        // Set cursor based on handle type
        handle.on("mouseenter", () => {
            if (!this.isSpacePressed) {
                let cursor = "pointer";
                if (
                    handleName.includes("top-left") ||
                    handleName.includes("bottom-right")
                ) {
                    cursor = "nwse-resize";
                } else if (
                    handleName.includes("top-right") ||
                    handleName.includes("bottom-left")
                ) {
                    cursor = "nesw-resize";
                } else if (
                    handleName.includes("top") ||
                    handleName.includes("bottom")
                ) {
                    cursor = "ns-resize";
                } else if (
                    handleName.includes("left") ||
                    handleName.includes("right")
                ) {
                    cursor = "ew-resize";
                }
                stage.container().style.cursor = cursor;
            }
        });

        handle.on("mouseleave", () => {
            if (!this.isSpacePressed) {
                stage.container().style.cursor = "default";
            }
        });

        // Handle dragging
        handle.on("dragmove", () => {
            this.resizeCropBox(handle);
        });
    };

    private resizeCropBox = (handle: Konva.Rect) => {
        if (!this.konva?.crop || !this.konva?.image) {
return;
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

        const minWidth = this.cropConstraints.minWidth ?? 64;
        const minHeight = this.cropConstraints.minHeight ?? 64;

        // Update dimensions based on handle type
        if (handleName.includes("left")) {
            const right = newX + newWidth;
            newX = Math.max(0, Math.min(handleX, right - minWidth));
            newWidth = right - newX;
        }
        if (handleName.includes("right")) {
            newWidth = Math.max(
                minWidth,
                Math.min(handleX - newX, imgWidth - newX),
            );
        }
        if (handleName.includes("top")) {
            const bottom = newY + newHeight;
            newY = Math.max(0, Math.min(handleY, bottom - minHeight));
            newHeight = bottom - newY;
        }
        if (handleName.includes("bottom")) {
            newHeight = Math.max(
                minHeight,
                Math.min(handleY - newY, imgHeight - newY),
            );
        }

        // Early boundary check for aspect ratio mode
        // If we're at a boundary and have aspect ratio, we need special handling
        if (this.cropConstraints.aspectRatio) {
            const atLeftEdge = rect.x() <= 0;
            const atRightEdge = rect.x() + rect.width() >= imgWidth;
            const atTopEdge = rect.y() <= 0;
            const atBottomEdge = rect.y() + rect.height() >= imgHeight;

            // For edge handles at boundaries, prevent invalid operations
            if (handleName === "left" && atLeftEdge && handleX >= rect.x()) {
                // Can't move left edge further left, only right (shrinking)
                return;
            }
            if (
                handleName === "right" &&
                atRightEdge &&
                handleX <= rect.x() + rect.width()
            ) {
                // Can't move right edge further right, only left (shrinking)
                return;
            }
            if (handleName === "top" && atTopEdge && handleY >= rect.y()) {
                // Can't move top edge further up, only down (shrinking)
                return;
            }
            if (
                handleName === "bottom" &&
                atBottomEdge &&
                handleY <= rect.y() + rect.height()
            ) {
                // Can't move bottom edge further down, only up (shrinking)
                return;
            }
        }

        // Apply constraints
        if (this.cropConstraints.maxWidth) {
            newWidth = Math.min(newWidth, this.cropConstraints.maxWidth);
        }
        if (this.cropConstraints.maxHeight) {
            newHeight = Math.min(newHeight, this.cropConstraints.maxHeight);
        }

        // Apply aspect ratio if set
        if (this.cropConstraints.aspectRatio) {
            const ratio = this.cropConstraints.aspectRatio;
            const oldX = rect.x();
            const oldY = rect.y();
            const oldWidth = rect.width();
            const oldHeight = rect.height();

            // Define anchor points (opposite of the handle being dragged)
            let anchorX = oldX;
            let anchorY = oldY;

            if (handleName.includes("right")) {
                anchorX = oldX; // Left edge is anchor
            } else if (handleName.includes("left")) {
                anchorX = oldX + oldWidth; // Right edge is anchor
            } else {
                anchorX = oldX + oldWidth / 2; // Center X is anchor for top/bottom
            }

            if (handleName.includes("bottom")) {
                anchorY = oldY; // Top edge is anchor
            } else if (handleName.includes("top")) {
                anchorY = oldY + oldHeight; // Bottom edge is anchor
            } else {
                anchorY = oldY + oldHeight / 2; // Center Y is anchor for left/right
            }

            // Calculate new dimensions maintaining aspect ratio
            if (handleName === "left" || handleName === "right") {
                // For left/right handles, adjust height to maintain ratio
                newHeight = newWidth / ratio;

                // Use center Y as anchor point
                newY = anchorY - newHeight / 2;
            } else if (handleName === "top" || handleName === "bottom") {
                // For top/bottom handles, adjust width to maintain ratio
                newWidth = newHeight * ratio;

                // Use center X as anchor point
                newX = anchorX - newWidth / 2;
            } else {
                // Corner handles - the anchor is the opposite corner
                // Use mouse position relative to anchor to determine constraint
                const mouseDistanceFromAnchorX = Math.abs(handleX - anchorX);
                const mouseDistanceFromAnchorY = Math.abs(handleY - anchorY);
                
                // Calculate maximum possible dimensions based on anchor position and image bounds
                let maxPossibleWidth; let maxPossibleHeight;
                
                if (handleName.includes("left")) {
                    // Anchor is on the right, max width is anchor X position
                    maxPossibleWidth = anchorX;
                } else {
                    // Anchor is on the left, max width is image width minus anchor X
                    maxPossibleWidth = imgWidth - anchorX;
                }
                
                if (handleName.includes("top")) {
                    // Anchor is on the bottom, max height is anchor Y position
                    maxPossibleHeight = anchorY;
                } else {
                    // Anchor is on the top, max height is image height minus anchor Y
                    maxPossibleHeight = imgHeight - anchorY;
                }
                
                // Constrain mouse distances to stay within image bounds
                const constrainedMouseDistanceX = Math.min(mouseDistanceFromAnchorX, maxPossibleWidth);
                const constrainedMouseDistanceY = Math.min(mouseDistanceFromAnchorY, maxPossibleHeight);

                // Determine which dimension should be the primary constraint
                // based on which direction the mouse moved further from the anchor (after constraining)
                const shouldConstrainByWidth = constrainedMouseDistanceX / ratio > constrainedMouseDistanceY;

                if (shouldConstrainByWidth) {
                    // Width is the primary dimension, calculate height from it
                    newWidth = constrainedMouseDistanceX;
                    newHeight = newWidth / ratio;
                    
                    // If calculated height exceeds bounds, switch to height constraint
                    if (newHeight > maxPossibleHeight) {
                        newHeight = maxPossibleHeight;
                        newWidth = newHeight * ratio;
                    }
                } else {
                    // Height is the primary dimension, calculate width from it
                    newHeight = constrainedMouseDistanceY;
                    newWidth = newHeight * ratio;
                    
                    // If calculated width exceeds bounds, switch to width constraint
                    if (newWidth > maxPossibleWidth) {
                        newWidth = maxPossibleWidth;
                        newHeight = newWidth / ratio;
                    }
                }

                // For corner handles, keep the opposite corner fixed
                if (handleName.includes("left")) {
                    newX = anchorX - newWidth;
                } else {
                    newX = anchorX;
                }

                if (handleName.includes("top")) {
                    newY = anchorY - newHeight;
                } else {
                    newY = anchorY;
                }
            }

            // Boundary checks and adjustments
            // Check if we exceed image bounds and need to adjust
            if (newX < 0) {
                const adjustment = -newX;
                newX = 0;
                // If we're anchored on the right, we need to adjust width
                if (handleName.includes("left")) {
                    newWidth -= adjustment;
                    newHeight = newWidth / ratio;
                    if (handleName !== "left") {
                        // For corner handles, also adjust Y to maintain anchor
                        newY = anchorY - newHeight;
                    }
                }
            }

            if (newY < 0) {
                const adjustment = -newY;
                newY = 0;
                // If we're anchored on the bottom, we need to adjust height
                if (handleName.includes("top")) {
                    newHeight -= adjustment;
                    newWidth = newHeight * ratio;
                    if (handleName !== "top") {
                        // For corner handles, also adjust X to maintain anchor
                        newX = anchorX - newWidth;
                    }
                }
            }

            if (newX + newWidth > imgWidth) {
                const adjustment = newX + newWidth - imgWidth;
                // If we're anchored on the left, we need to adjust width
                if (handleName.includes("right")) {
                    newWidth -= adjustment;
                    newHeight = newWidth / ratio;
                    if (handleName !== "right") {
                        // For corner handles, maintain anchor
                        newY = anchorY - newHeight;
                    }
                } else if (handleName === "top" || handleName === "bottom") {
                    // For vertical handles, recenter
                    newX = imgWidth - newWidth;
                    if (newX < 0) {
                        newWidth = imgWidth;
                        newHeight = newWidth / ratio;
                        newX = 0;
                    }
                }
            }

            if (newY + newHeight > imgHeight) {
                const adjustment = newY + newHeight - imgHeight;
                // If we're anchored on the top, we need to adjust height
                if (handleName.includes("bottom")) {
                    newHeight -= adjustment;
                    newWidth = newHeight * ratio;
                    if (handleName !== "bottom") {
                        // For corner handles, maintain anchor
                        newX = anchorX - newWidth;
                    }
                } else if (handleName === "left" || handleName === "right") {
                    // For horizontal handles, recenter
                    newY = imgHeight - newHeight;
                    if (newY < 0) {
                        newHeight = imgHeight;
                        newWidth = newHeight * ratio;
                        newY = 0;
                    }
                }
            }

            // Final check for minimum sizes
            if (newWidth < minWidth) {
                newWidth = minWidth;
                newHeight = newWidth / ratio;
                // Reposition based on anchor
                if (handleName.includes("left")) {
                    newX = anchorX - newWidth;
                }
                if (handleName.includes("top")) {
                    newY = anchorY - newHeight;
                }
            }
            if (newHeight < minHeight) {
                newHeight = minHeight;
                newWidth = newHeight * ratio;
                // Reposition based on anchor
                if (handleName.includes("left")) {
                    newX = anchorX - newWidth;
                }
                if (handleName.includes("top")) {
                    newY = anchorY - newHeight;
                }
            }
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

        this.callbacks.onCropChange?.({
            x: newX,
            y: newY,
            width: newWidth,
            height: newHeight,
        });
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

        if (handleName.includes("right")) {
x += rect.width();
} else if (handleName.includes("left")) {
x += 0;
} else {
x += rect.width() / 2;
}

        if (handleName.includes("bottom")) {
y += rect.height();
} else if (handleName.includes("top")) {
y += 0;
} else {
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
            line1.points([
                x + verticalThird * 2,
                y,
                x + verticalThird * 2,
                y + height,
            ]);
        }

        // Update horizontal lines
        const horizontalThird = height / 3;
        const line2 = lines[2];
        const line3 = lines[3];
        if (line2 instanceof Konva.Line) {
            line2.points([
                x,
                y + horizontalThird,
                x + width,
                y + horizontalThird,
            ]);
        }
        if (line3 instanceof Konva.Line) {
            line3.points([
                x,
                y + horizontalThird * 2,
                x + width,
                y + horizontalThird * 2,
            ]);
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
        const handleSize = 8 / scale;
        const strokeWidth = 1 / scale;

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

    cancelCrop = () => {
        if (!this.isInCropMode || !this.konva?.crop) {
return;
}

        this.isInCropMode = false;
        this.konva.crop.layer.destroy();
        this.konva.crop = undefined;
    };

    applyCrop = () => {
        if (!this.isInCropMode || !this.konva?.crop) {
return;
}

        const rect = this.konva.crop.rect;

        // If there's already an applied crop, combine them
        if (this.appliedCrop) {
            // The new crop is relative to the already cropped image
            this.appliedCrop = {
                x: this.appliedCrop.x + rect.x(),
                y: this.appliedCrop.y + rect.y(),
                width: rect.width(),
                height: rect.height(),
            };
        } else {
            this.appliedCrop = {
                x: rect.x(),
                y: rect.y(),
                width: rect.width(),
                height: rect.height(),
            };
        }

        this.cancelCrop();

        // Redisplay image with crop applied
        this.displayImage();
    };

    resetCrop = () => {
        this.appliedCrop = undefined;

        // Redisplay image without crop
        this.displayImage();
    };

    hasCrop = (): boolean => {
        return !!this.appliedCrop;
    };

    // Export
    exportImage = async (
        format: "canvas" | "blob" | "dataURL" = "blob",
    ): Promise<HTMLCanvasElement | Blob | string> => {
        if (!this.originalImage) {
            throw new Error("No image loaded");
        }

        // Create temporary canvas
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            throw new Error("Failed to get canvas context");
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
                    this.appliedCrop.height,
                );
            } else {
                canvas.width = this.originalImage.width;
                canvas.height = this.originalImage.height;
                ctx.drawImage(this.originalImage, 0, 0);
            }

            if (format === "canvas") {
                return canvas;
            } else if (format === "dataURL") {
                try {
                    return canvas.toDataURL("image/png");
                } catch (error) {
                    throw new Error("Cannot export image: Canvas is tainted by cross-origin data. Try loading the image from the same domain or use a CORS-enabled source.");
                }
            } else {
                return new Promise((resolve, reject) => {
                    try {
                        canvas.toBlob((blob) => {
                            if (blob) {
                                resolve(blob);
                            } else {
                                reject(new Error("Failed to create blob"));
                            }
                        }, "image/png");
                    } catch (error) {
                        reject(new Error("Cannot export image: Canvas is tainted by cross-origin data. Try loading the image from the same domain or use a CORS-enabled source."));
                    }
                });
            }
        } catch (error) {
            if (error instanceof Error && error.message.includes("tainted")) {
                throw new Error("Cannot export image: Canvas is tainted by cross-origin data. Try loading the image from the same domain or use a CORS-enabled source.");
            }
            throw error;
        }
    };

    // View Control
    setZoom = (scale: number, point?: { x: number; y: number }) => {
        if (!this.konva) {
return;
}

        scale = Math.max(this.zoomMin, Math.min(this.zoomMax, scale));

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

        this.callbacks.onZoomChange?.(scale);
    };

    getZoom = (): number => {
        return this.konva?.stage.scaleX() || 1;
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

        const scale =
            Math.min(
                containerWidth / imageWidth,
                containerHeight / imageHeight,
            ) * 0.9; // 90% to add some padding

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

        this.callbacks.onZoomChange?.(scale);
    };

    // Configuration
    setCallbacks = (callbacks: EditorCallbacks) => {
        this.callbacks = { ...this.callbacks, ...callbacks };
    };

    setCropAspectRatio = (ratio: number | undefined) => {
        // Update the constraint
        this.cropConstraints.aspectRatio = ratio;

        // If we're currently cropping, adjust the crop box to match the new ratio
        if (this.isInCropMode && this.konva?.crop && this.konva?.image) {
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
            const minWidth = this.cropConstraints.minWidth ?? 64;
            const minHeight = this.cropConstraints.minHeight ?? 64;

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
            this.callbacks.onCropChange?.({
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
    };

    destroy = () => {
        // Remove window event listeners
        if (this.keydownHandler) {
            window.removeEventListener("keydown", this.keydownHandler);
            this.keydownHandler = undefined;
        }
        if (this.keyupHandler) {
            window.removeEventListener("keyup", this.keyupHandler);
            this.keyupHandler = undefined;
        }

        // Remove stage container event listeners
        if (this.konva) {
            const container = this.konva.stage.container();
            if (this.contextMenuHandler) {
                container.removeEventListener(
                    "contextmenu",
                    this.contextMenuHandler,
                );
                this.contextMenuHandler = undefined;
            }
            if (this.wheelHandler) {
                container.removeEventListener("wheel", this.wheelHandler);
                this.wheelHandler = undefined;
            }
        }

        // Clean up blob URL if it exists
        if (this.currentImageBlobUrl) {
            URL.revokeObjectURL(this.currentImageBlobUrl);
            this.currentImageBlobUrl = undefined;
        }

        // Cancel any ongoing crop operation
        if (this.isInCropMode) {
            this.cancelCrop();
        }

        // Remove all Konva event listeners by destroying the stage
        // This automatically removes all Konva event handlers
        this.konva?.stage.destroy();

        // Clear all references
        this.konva = undefined;
        this.originalImage = undefined;
        this.appliedCrop = undefined;
        this.callbacks = {};
    };
}