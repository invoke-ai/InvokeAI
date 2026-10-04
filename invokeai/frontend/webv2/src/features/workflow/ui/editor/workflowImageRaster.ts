export type WorkflowRasterOptions = {
  backgroundColor: string;
  height: number;
  /**
   * Checked between stages only. html-to-image must never see it: it caches whatever a failed fetch produced, so an
   * aborted font fetch would leave every later capture without the font until reload.
   */
  signal: AbortSignal;
  /** Computed style properties copied onto the serialized clone. */
  styleProperties: readonly string[];
  width: number;
};

/**
 * Settles on every outcome. html-to-image's own loader waits on `decode()` without handling its rejection, so a page
 * that loads but cannot decode would leave its capture pending forever.
 */
const loadSvgImage = (url: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      image.decode().then(
        () => resolve(image),
        () => reject(new Error('Workflow image could not be decoded'))
      );
    };
    image.onerror = () => reject(new Error('Workflow image could not be loaded'));
    image.decoding = 'async';
    image.src = url;
  });

const canvasToPng = (canvas: HTMLCanvasElement) =>
  new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, 'image/png');
  });

/**
 * Renders `element` 1:1 into a `width` x `height` PNG. The SVG page, the canvas and the PNG all have exactly the
 * requested size, so the caller's size budget bounds every raster surface this allocates. Images must already be
 * inlined as data URLs; html-to-image would otherwise fetch and cache them for the rest of the session.
 */
export const rasterizeWorkflowImage = async (
  element: HTMLElement,
  { backgroundColor, height, signal, styleProperties, width }: WorkflowRasterOptions
): Promise<Blob> => {
  // Loaded on first use: only an export needs the serializer.
  const { toSvg } = await import('html-to-image');
  const svg = await toSvg(element, {
    backgroundColor,
    height,
    includeStyleProperties: [...styleProperties],
    skipFonts: false,
    width,
  });
  signal.throwIfAborted();
  const image = await loadSvgImage(svg);
  const canvas = document.createElement('canvas');
  try {
    signal.throwIfAborted();
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Workflow image canvas is unavailable');
    }
    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    const blob = await canvasToPng(canvas);
    if (!blob) {
      throw new Error('Workflow image export returned an empty Blob');
    }
    return blob;
  } finally {
    // Release the backing store and the serialized page now rather than whenever these are collected.
    canvas.width = 0;
    canvas.height = 0;
    image.src = '';
  }
};
