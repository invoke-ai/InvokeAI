import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  CanvasEraserLineState,
  CanvasEraserLineWithPressureState,
  CanvasImageState,
  CanvasRectState,
  RgbaColor,
} from 'features/controlLayers/store/types';

/**
 * Options for converting bitmap to mask objects
 */
export interface BitmapToMaskOptions {
  /**
   * The threshold for considering a pixel as masked (0-255)
   * Pixels with alpha >= threshold are considered masked
   */
  threshold?: number;
  /**
   * The color to use for brush lines
   */
  brushColor?: RgbaColor;
  /**
   * The stroke width for brush lines
   */
  strokeWidth?: number;
  /**
   * Whether to use pressure-sensitive lines
   */
  usePressure?: boolean;
  /**
   * The pressure value to use for pressure-sensitive lines (0-1)
   */
  pressure?: number;
}

/**
 * Default options for bitmap to mask conversion
 */
const DEFAULT_OPTIONS: Required<BitmapToMaskOptions> = {
  threshold: 128,
  brushColor: { r: 255, g: 255, b: 255, a: 1 },
  strokeWidth: 50,
  usePressure: false,
  pressure: 1.0,
};

/**
 * Converts a bitmap (ImageData) to mask objects (brush lines, eraser lines, rectangles)
 *
 * @param imageData - The bitmap data to convert
 * @param options - Conversion options
 * @returns Array of mask objects
 */
export function bitmapToMaskObjects(
  imageData: ImageData,
  options: BitmapToMaskOptions = {}
): (
  | CanvasBrushLineState
  | CanvasBrushLineWithPressureState
  | CanvasEraserLineState
  | CanvasEraserLineWithPressureState
  | CanvasRectState
  | CanvasImageState
)[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const { width, height, data } = imageData;
  const objects: (
    | CanvasBrushLineState
    | CanvasBrushLineWithPressureState
    | CanvasEraserLineState
    | CanvasEraserLineWithPressureState
    | CanvasRectState
    | CanvasImageState
  )[] = [];

  // For now, we'll create a simple approach that creates rectangles for masked areas
  // This can be enhanced later to create more sophisticated brush/eraser line patterns

  // Scan the image data to find masked areas
  for (let y = 0; y < height; y += opts.strokeWidth) {
    for (let x = 0; x < width; x += opts.strokeWidth) {
      // Check if this pixel is masked
      const pixelIndex = (y * width + x) * 4;
      const alpha = data[pixelIndex + 3] ?? 0;

      if (alpha >= opts.threshold) {
        // Create a rectangle for this masked area
        const rect: CanvasRectState = {
          id: getPrefixedId('rect'),
          type: 'rect',
          rect: {
            x,
            y,
            width: Math.min(opts.strokeWidth, width - x),
            height: Math.min(opts.strokeWidth, height - y),
          },
          color: opts.brushColor,
        };
        objects.push(rect);
      }
    }
  }

  return objects;
}

/**
 * Inverts a bitmap by flipping the alpha channel
 *
 * @param imageData - The bitmap data to invert
 * @returns New ImageData with inverted alpha channel
 */
export function invertBitmap(imageData: ImageData): ImageData {
  const { width, height, data } = imageData;
  const newImageData = new ImageData(width, height);
  const newData = newImageData.data;

  for (let i = 0; i < data.length; i += 4) {
    // Copy RGB values
    newData[i] = data[i] ?? 0; // R
    newData[i + 1] = data[i + 1] ?? 0; // G
    newData[i + 2] = data[i + 2] ?? 0; // B
    // Invert alpha
    newData[i + 3] = 255 - (data[i + 3] ?? 0); // A
  }

  return newImageData;
}

/**
 * Converts mask objects to a bitmap (ImageData)
 * This is a simplified version that creates a basic bitmap representation
 *
 * @param objects - Array of mask objects
 * @param width - Width of the output bitmap
 * @param height - Height of the output bitmap
 * @returns ImageData representing the mask
 */
export function maskObjectsToBitmap(
  objects: (
    | CanvasBrushLineState
    | CanvasBrushLineWithPressureState
    | CanvasEraserLineState
    | CanvasEraserLineWithPressureState
    | CanvasRectState
    | CanvasImageState
  )[],
  width: number,
  height: number
): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Failed to get canvas context');
  }

  // Clear canvas with transparent background
  ctx.clearRect(0, 0, width, height);

  // Draw each object
  for (const obj of objects) {
    if (obj.type === 'rect') {
      ctx.fillStyle = `rgba(${obj.color.r}, ${obj.color.g}, ${obj.color.b}, ${obj.color.a})`;
      ctx.fillRect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    } else if (obj.type === 'brush_line' || obj.type === 'brush_line_with_pressure') {
      ctx.strokeStyle = `rgba(${obj.color.r}, ${obj.color.g}, ${obj.color.b}, ${obj.color.a})`;
      ctx.lineWidth = obj.strokeWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Draw the line
      ctx.beginPath();
      for (let i = 0; i < obj.points.length; i += 2) {
        const x = obj.points[i] ?? 0;
        const y = obj.points[i + 1] ?? 0;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    } else if (obj.type === 'eraser_line' || obj.type === 'eraser_line_with_pressure') {
      // Eraser lines use destination-out composite operation
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
      ctx.lineWidth = obj.strokeWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Draw the line
      ctx.beginPath();
      for (let i = 0; i < obj.points.length; i += 2) {
        const x = obj.points[i] ?? 0;
        const y = obj.points[i + 1] ?? 0;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Reset composite operation
      ctx.globalCompositeOperation = 'source-over';
    } else if (obj.type === 'image') {
      // For image objects, we need to load the image and draw it
      // This is a simplified approach - in a real implementation, you'd want to handle image loading properly
      const img = new Image();
      img.crossOrigin = 'anonymous';

      // Create a temporary canvas to draw the image
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx) {
        tempCanvas.width = obj.image.width;
        tempCanvas.height = obj.image.height;

        // Draw the image to the temp canvas
        if ('image_name' in obj.image) {
          // This would need proper image loading from the server
          // For now, we'll skip image objects in the mask conversion
          console.warn('Image objects with image_name are not supported in mask conversion');
        } else {
          // Data URL image
          img.src = obj.image.dataURL;
          tempCtx.drawImage(img, 0, 0);

          // Draw the temp canvas to the main canvas
          ctx.drawImage(tempCanvas, 0, 0);
        }
      }
    }
  }

  return ctx.getImageData(0, 0, width, height);
}
