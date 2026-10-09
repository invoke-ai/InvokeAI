import type { Rect } from '@xyflow/react';

import { downloadBlob } from '@platform/browser/downloadBlob';

import { rasterizeWorkflowImage } from './workflowImageRaster';

const WORKFLOW_GRID_SIZE = 25;

export const EXPORT_PADDING = 100;
export const EXPORT_SCALE = 2;
export const WORKFLOW_EXPORT_TIMEOUT_MS = 30_000;
export const WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS = 5_000;
export const WORKFLOW_EXPORT_LAYOUT_TIMEOUT_MS = 5_000;
// A capture cannot be interrupted, so bound how many may still be running after their callers timed out.
const MAX_ACTIVE_WORKFLOW_RASTERIZATIONS = 2;
let activeWorkflowRasterizations = 0;

export type WorkflowImageExportLimits = {
  /** Output pixels (width x height). */
  maxPixels: number;
  /** Output pixels on either side. */
  maxSide: number;
  /** Below this many output pixels per workflow pixel the export is refused rather than shrunk further. */
  minScale: number;
};

/**
 * Application headroom, not a browser guarantee: engines reject canvases above 268,435,456 px (Chromium, desktop
 * WebKit), 67,108,864 px (iOS WebKit) or 65,535 px per side, and can fail to allocate well below that. A capture holds
 * several output-sized RGBA surfaces at once (the decoded SVG page, the canvas and its PNG snapshot); 32 MiP is half
 * the smallest engine area check and keeps each surface at 128 MiB. 16,384 px per side is a conservative application
 * choice, html-to-image's own heuristic, well under the engines' side checks. At 0.5x a workflow reads like the editor
 * at 50% zoom; below that field text stops being legible, so the image would no longer document the workflow.
 */
const WORKFLOW_IMAGE_EXPORT_LIMITS: WorkflowImageExportLimits = {
  maxPixels: 2 ** 25,
  maxSide: 16_384,
  minScale: 0.5,
};

export type WorkflowImageExportPlan = {
  /** Padded workflow size in layout pixels. */
  width: number;
  height: number;
  /** Output pixels per layout pixel: `EXPORT_SCALE` unless the budget requires less. */
  scale: number;
  outputWidth: number;
  outputHeight: number;
};

export type WorkflowImageExportOutcome =
  /** `reduced`: the image is smaller than the editor's 1:1 layout, which is worth telling the user. */
  | { status: 'exported'; reduced: boolean; width: number; height: number }
  | { status: 'busy' | 'canceled' | 'too-large' };

export const EXPORT_STYLE_PROPERTIES = [
  'box-sizing',
  'display',
  'position',
  'inset',
  'top',
  'right',
  'bottom',
  'left',
  'width',
  'height',
  'min-width',
  'min-height',
  'max-width',
  'max-height',
  'padding',
  'padding-top',
  'padding-right',
  'padding-bottom',
  'padding-left',
  'margin',
  'margin-top',
  'margin-right',
  'margin-bottom',
  'margin-left',
  'flex',
  'flex-direction',
  'flex-wrap',
  'aspect-ratio',
  'flex-grow',
  'flex-shrink',
  'flex-basis',
  'align-items',
  'align-content',
  'align-self',
  'justify-content',
  'gap',
  'row-gap',
  'column-gap',
  'grid-template-columns',
  'grid-template-rows',
  'grid-column',
  'grid-row',
  'overflow',
  'overflow-x',
  'overflow-y',
  'visibility',
  'opacity',
  'transform',
  'transform-origin',
  'color',
  'background',
  'background-color',
  'background-image',
  'background-size',
  'background-position',
  'background-repeat',
  'background-clip',
  'background-origin',
  'border',
  'border-width',
  'border-style',
  'border-color',
  'border-radius',
  'box-shadow',
  'font-family',
  'font-size',
  'font-weight',
  'font-style',
  'direction',
  'line-height',
  'letter-spacing',
  'text-align',
  'text-overflow',
  'text-decoration',
  'text-transform',
  'text-shadow',
  '-webkit-line-clamp',
  '-webkit-box-orient',
  'white-space',
  'word-break',
  'overflow-wrap',
  'filter',
  'object-fit',
  'object-position',
  'fill',
  'stroke',
  'stroke-width',
  'stroke-linecap',
  'stroke-linejoin',
  'stroke-dasharray',
  'stroke-dashoffset',
  'z-index',
  'pointer-events',
  'vertical-align',
] as const;
export const SVG_EXPORT_STYLE_PROPERTIES = [
  ...EXPORT_STYLE_PROPERTIES,
  'fill-opacity',
  'stroke-opacity',
  'marker-start',
  'marker-mid',
  'marker-end',
  'paint-order',
  'shape-rendering',
  'vector-effect',
  'clip-path',
  'mask',
] as const;

type WorkflowContentBoundsOptions = {
  includeInputFieldLabels?: boolean;
};

export const getWorkflowContentBounds = (
  flowElement: HTMLElement,
  nodeBounds: Rect,
  { includeInputFieldLabels = true }: WorkflowContentBoundsOptions = {}
): Rect => {
  const flowRect = flowElement.getBoundingClientRect();
  const viewport = flowElement.querySelector<HTMLElement>('.react-flow__viewport');
  const viewportRect = viewport?.getBoundingClientRect() ?? flowRect;
  const transform = viewport ? (getComputedStyle(viewport).transform ?? 'none') : 'none';
  const matrix = transform
    .match(/^matrix\(([^)]+)\)$/)?.[1]
    ?.split(',')
    .map(Number);
  const zoom = matrix?.[0] && Number.isFinite(matrix[0]) && matrix[0] > 0 ? matrix[0] : 1;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  // The editor's measured nodes may be taller because of execution previews or textarea resize state.
  flowElement.querySelectorAll<HTMLElement>('.react-flow__node').forEach((node) => {
    const rect = node.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }
    const x = (rect.left - viewportRect.left) / zoom;
    const y = (rect.top - viewportRect.top) / zoom;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x + rect.width / zoom);
    maxY = Math.max(maxY, y + rect.height / zoom);
  });
  if (minX === Infinity) {
    minX = nodeBounds.x;
    minY = nodeBounds.y;
    maxX = nodeBounds.x + nodeBounds.width;
    maxY = nodeBounds.y + nodeBounds.height;
  }

  flowElement.querySelectorAll<SVGGraphicsElement>('.react-flow__edge-path').forEach((path) => {
    let pathBounds: DOMRect;
    try {
      pathBounds = path.getBBox();
    } catch {
      return;
    }
    if (
      !Number.isFinite(pathBounds.x) ||
      !Number.isFinite(pathBounds.y) ||
      !Number.isFinite(pathBounds.width) ||
      !Number.isFinite(pathBounds.height)
    ) {
      return;
    }

    minX = Math.min(minX, pathBounds.x);
    minY = Math.min(minY, pathBounds.y);
    maxX = Math.max(maxX, pathBounds.x + pathBounds.width);
    maxY = Math.max(maxY, pathBounds.y + pathBounds.height);
  });

  {
    const contentElements = new Set<HTMLElement>([
      ...(includeInputFieldLabels
        ? flowElement.querySelectorAll<HTMLElement>('[data-node-input-field-title="true"]')
        : []),
      ...flowElement.querySelectorAll<HTMLElement>('[data-workflow-export-field-content="true"]'),
      ...flowElement.querySelectorAll<HTMLElement>('[data-workflow-export-static-node-content="true"]'),
      ...flowElement.querySelectorAll<HTMLElement>('[data-workflow-export-output-title="true"]'),
    ]);
    contentElements.forEach((element) => {
      const elementRect = element.getBoundingClientRect();
      const intrinsicWidth = Math.max(elementRect.width, element.scrollWidth * zoom);
      const intrinsicHeight = Math.max(elementRect.height, element.scrollHeight * zoom);
      const overflowWidth = Math.max(0, intrinsicWidth - elementRect.width) / zoom;
      const direction = getComputedStyle(element).direction;
      const elementX = (elementRect.left - viewportRect.left) / zoom - (direction === 'rtl' ? overflowWidth : 0);
      const elementY = (elementRect.top - viewportRect.top) / zoom;
      const labelWidth = intrinsicWidth / zoom;
      const labelHeight = intrinsicHeight / zoom;

      minX = Math.min(minX, elementX);
      minY = Math.min(minY, elementY);
      maxX = Math.max(maxX, elementX + labelWidth);
      maxY = Math.max(maxY, elementY + labelHeight);
    });
  }

  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
};

/**
 * Chooses the largest scale up to `EXPORT_SCALE` that keeps the output within `limits`, or null when even
 * `limits.minScale` does not fit. The capture renders its SVG page and canvas at the output size, so this bounds every
 * raster surface; the serialized SVG data URL grows with the DOM and embedded images instead, and is not bounded here.
 */
export const planWorkflowImageExport = (
  bounds: Rect,
  limits: WorkflowImageExportLimits = WORKFLOW_IMAGE_EXPORT_LIMITS
): WorkflowImageExportPlan | null => {
  const width = Math.max(1, Math.ceil(bounds.width + EXPORT_PADDING * 2));
  const height = Math.max(1, Math.ceil(bounds.height + EXPORT_PADDING * 2));
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new RangeError('Workflow image bounds are not finite');
  }

  const scale = Math.min(
    EXPORT_SCALE,
    limits.maxSide / width,
    limits.maxSide / height,
    Math.sqrt(limits.maxPixels / (width * height))
  );
  if (scale < limits.minScale) {
    return null;
  }

  // Rounding down keeps both sides and their product within the limits.
  return {
    width,
    height,
    scale,
    outputWidth: Math.max(1, Math.floor(width * scale)),
    outputHeight: Math.max(1, Math.floor(height * scale)),
  };
};

export const getWorkflowExportCloneStyle = (plan: WorkflowImageExportPlan) => ({
  width: `${plan.width}px`,
  height: `${plan.height}px`,
  position: 'relative',
  left: '0',
  top: '0',
  pointerEvents: 'none',
});

/** Lays the clone out at workflow size and paints it at output size, so scaling never reflows labels. */
const getWorkflowExportCaptureStyles = (plan: WorkflowImageExportPlan) => ({
  root: { width: `${plan.outputWidth}px`, height: `${plan.outputHeight}px`, overflow: 'hidden' },
  clone: { transform: `scale(${plan.scale})`, transformOrigin: '0 0' },
});

export const getWorkflowSvgExportStyles = (computedStyle: Pick<CSSStyleDeclaration, 'getPropertyValue'>) =>
  SVG_EXPORT_STYLE_PROPERTIES.reduce<Record<string, string>>((styles, property) => {
    const value = computedStyle.getPropertyValue(property);
    if (value) {
      styles[property] = value;
    }
    return styles;
  }, {});

export const sanitizeWorkflowImageFilename = (workflowName: string, fallbackWorkflowName: string): string => {
  const sanitizedName = workflowName
    .replace(/[<>:"/\\|?*]/g, '-')
    .split('')
    .map((character) => (character.charCodeAt(0) < 32 ? '-' : character))
    .join('')
    .trim()
    .replace(/[. ]+$/g, '');

  return sanitizedName || fallbackWorkflowName;
};

const setExportElementStyle = (element: HTMLElement | SVGElement, property: string, value: string) => {
  element.style.setProperty(property, value, 'important');
};

export const getWorkflowExportStagingStyle = (plan: WorkflowImageExportPlan) => ({
  position: 'fixed',
  left: '-100000px',
  top: '0',
  width: `${plan.width}px`,
  height: `${plan.height}px`,
  pointerEvents: 'none',
});

const setBackgroundGridForExport = (root: HTMLElement, translation: { x: number; y: number }) => {
  const background = root.querySelector<SVGSVGElement>('.react-flow__background');
  const pattern = background?.querySelector<SVGPatternElement>('pattern');
  if (!background || !pattern) {
    return;
  }

  const patternId = pattern.id.endsWith('-export') ? pattern.id : `${pattern.id}-export`;
  pattern.id = patternId;
  pattern.setAttribute('width', `${WORKFLOW_GRID_SIZE}`);
  pattern.setAttribute('height', `${WORKFLOW_GRID_SIZE}`);
  pattern.setAttribute('x', `${((translation.x % WORKFLOW_GRID_SIZE) + WORKFLOW_GRID_SIZE) % WORKFLOW_GRID_SIZE}`);
  pattern.setAttribute('y', `${((translation.y % WORKFLOW_GRID_SIZE) + WORKFLOW_GRID_SIZE) % WORKFLOW_GRID_SIZE}`);
  pattern.setAttribute('patternTransform', `translate(-${WORKFLOW_GRID_SIZE},-${WORKFLOW_GRID_SIZE})`);

  const patternReference = `url(#${patternId})`;
  background.querySelector('rect')?.setAttribute('fill', patternReference);

  const dot = pattern.querySelector('circle');
  dot?.setAttribute('cx', '0.5');
  dot?.setAttribute('cy', '0.5');
  dot?.setAttribute('r', '0.5');
};

const inlineSvgStylesForExport = (root: HTMLElement) => {
  root
    .querySelectorAll<SVGElement>(
      '.react-flow__edges svg, .react-flow__edges svg *, .react-flow__background, .react-flow__background *'
    )
    .forEach((element) => {
      const styles = getWorkflowSvgExportStyles(getComputedStyle(element));
      Object.entries(styles).forEach(([property, value]) => {
        element.style.setProperty(property, value, 'important');
      });
    });
};

export const setWorkflowExportNodeOpacity = (root: HTMLElement) => {
  root.querySelectorAll<HTMLElement>('.react-flow__node > [data-is-selected]').forEach((element) => {
    setExportElementStyle(element, 'opacity', '1');
  });
  root.querySelectorAll<HTMLElement>('.react-flow__node').forEach((element) => {
    setExportElementStyle(element, 'opacity', '1');
  });
};

export const hideWorkflowExportStatusIndicators = (root: HTMLElement) => {
  root.querySelectorAll<HTMLElement>('[data-node-status-indicator="true"]').forEach((element) => {
    setExportElementStyle(element, 'display', 'none');
  });
};

export const hideWorkflowExportInfoIcons = (root: HTMLElement) => {
  root.querySelectorAll<SVGElement>('[data-node-info-icon="true"]').forEach((element) => {
    setExportElementStyle(element, 'display', 'none');
  });
};

export const setWorkflowExportInputFieldTitleStyles = (root: HTMLElement) => {
  root.querySelectorAll<HTMLElement>('[data-node-input-field-title="true"]').forEach((element) => {
    setExportElementStyle(element, 'display', 'block');
    setExportElementStyle(element, 'white-space', 'nowrap');
    setExportElementStyle(element, 'overflow', 'visible');
    setExportElementStyle(element, 'text-overflow', 'clip');
  });
};

const namespaceWorkflowExportIds = (clone: HTMLElement) => {
  const elements = [clone, ...clone.querySelectorAll<HTMLElement | SVGElement>('*')];
  const ids = new Map<string, string>();

  elements.forEach((element) => {
    if (element.id) {
      ids.set(element.id, `${element.id}-workflow-export`);
    }
  });

  elements.forEach((element) => {
    const id = element.id;
    if (id) {
      element.id = ids.get(id) ?? id;
    }
  });

  const sortedIds = [...ids.entries()].sort(([first], [second]) => second.length - first.length);
  const rewriteReferences = (value: string) =>
    sortedIds.reduce((rewritten, [id, namespacedId]) => rewritten.split(`#${id}`).join(`#${namespacedId}`), value);

  elements.forEach((element) => {
    Array.from(element.attributes).forEach((attribute) => {
      const rewritten = rewriteReferences(attribute.value);
      if (rewritten !== attribute.value) {
        element.setAttribute(attribute.name, rewritten);
      }
    });
  });
};

const prepareExportClone = (clone: HTMLElement, bounds: Rect, plan: WorkflowImageExportPlan) => {
  const root = clone.matches('.react-flow') ? clone : clone.querySelector<HTMLElement>('.react-flow');
  const viewport = clone.querySelector<HTMLElement>('.react-flow__viewport');
  if (!root || !viewport) {
    throw new Error('Workflow editor DOM is missing React Flow viewport');
  }

  const translation = {
    x: EXPORT_PADDING - bounds.x,
    y: EXPORT_PADDING - bounds.y,
  };

  Object.assign(clone.style, getWorkflowExportCloneStyle(plan));

  root.style.width = `${plan.width}px`;
  root.style.height = `${plan.height}px`;
  setExportElementStyle(root, 'background-color', 'var(--xy-background-color)');
  viewport.style.transform = `translate(${translation.x}px, ${translation.y}px) scale(1)`;
  setBackgroundGridForExport(root, translation);

  clone
    .querySelectorAll<HTMLElement>('[data-is-selected], [data-selected], [data-are-connected-nodes-selected]')
    .forEach((element) => {
      if (element.hasAttribute('data-is-selected')) {
        element.setAttribute('data-is-selected', 'false');
      }
      if (element.hasAttribute('data-selected')) {
        element.setAttribute('data-selected', 'false');
      }
      if (element.hasAttribute('data-are-connected-nodes-selected')) {
        element.setAttribute('data-are-connected-nodes-selected', 'false');
      }
    });
  clone.querySelectorAll('.react-flow__node.selected, .react-flow__edge.selected').forEach((element) => {
    element.classList.remove('selected');
  });
  clone.querySelectorAll('.react-flow__edge.workflow-selected-node-edge').forEach((element) => {
    element.classList.remove('workflow-selected-node-edge');
  });
  clone
    .querySelectorAll<HTMLElement>(
      '[data-workflow-export-control="true"], .react-flow__controls, .react-flow__minimap, .react-flow__panel'
    )
    .forEach((element) => {
      setExportElementStyle(element, 'display', 'none');
    });
  clone.querySelectorAll<HTMLElement>('[data-workflow-node-shell="true"]').forEach((element) => {
    setExportElementStyle(element, 'border-color', 'var(--chakra-colors-border-emphasized)');
    setExportElementStyle(element, 'box-shadow', 'var(--chakra-shadows-sm)');
  });
  clone.querySelectorAll<HTMLElement>('[data-connector-node-body="true"]').forEach((element) => {
    setExportElementStyle(element, 'background-color', 'var(--chakra-colors-bg-emphasized)');
  });
  clone.querySelectorAll<HTMLElement>('[data-connector-node-icon="true"]').forEach((element) => {
    setExportElementStyle(element, 'color', 'var(--chakra-colors-fg)');
  });
  setWorkflowExportNodeOpacity(clone);
  hideWorkflowExportStatusIndicators(clone);
  hideWorkflowExportInfoIcons(clone);
  setWorkflowExportInputFieldTitleStyles(clone);

  clone
    .querySelectorAll<HTMLElement>('.react-flow__edges, .react-flow__edges > svg, .react-flow__edge')
    .forEach((element) => {
      setExportElementStyle(element, 'z-index', '0');
    });
  clone
    .querySelectorAll<HTMLElement>('.react-flow__edgelabel-renderer, .react-flow__edgelabel-renderer *')
    .forEach((element) => {
      setExportElementStyle(element, 'z-index', '0');
    });
  clone.querySelectorAll<HTMLElement>('.react-flow__nodes, .react-flow__node').forEach((element) => {
    setExportElementStyle(element, 'z-index', '1');
  });
  clone.querySelectorAll<HTMLElement>('.react-flow__selection, .react-flow__nodesselection').forEach((element) => {
    setExportElementStyle(element, 'display', 'none');
  });
};

/** Holds a rasterization slot until the capture itself settles, which can be long after its caller timed out. */
const rasterizeWithTimeout = async (
  captureRoot: HTMLElement,
  plan: WorkflowImageExportPlan,
  backgroundColor: string
): Promise<Blob> => {
  const controller = new AbortController();
  activeWorkflowRasterizations += 1;
  const releaseRasterizationSlot = () => {
    activeWorkflowRasterizations -= 1;
  };
  const capture = rasterizeWorkflowImage(captureRoot, {
    backgroundColor,
    height: plan.outputHeight,
    signal: controller.signal,
    styleProperties: EXPORT_STYLE_PROPERTIES,
    width: plan.outputWidth,
  });
  void capture.then(releaseRasterizationSlot, releaseRasterizationSlot);

  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      capture,
      new Promise<never>((_, reject) => {
        timeoutId = setTimeout(() => {
          // Skips the capture's remaining stages; work already running cannot be interrupted.
          controller.abort();
          reject(new Error(`Workflow image export timed out after ${WORKFLOW_EXPORT_TIMEOUT_MS} ms`));
        }, WORKFLOW_EXPORT_TIMEOUT_MS);
      }),
    ]);
  } finally {
    clearTimeout(timeoutId);
  }
};

const downloadPng = (blob: Blob, workflowName: string, fallbackWorkflowName: string) => {
  downloadBlob(blob, `${sanitizeWorkflowImageFilename(workflowName, fallbackWorkflowName)}.png`);
};

/**
 * React Flow keeps a freshly mounted node hidden until it has measured it, so a capture taken earlier omits it. The
 * export view mounts just before capture, and measurement waits for a rendered frame. A zero-size node is never
 * measured and paints nothing, and a removed node no longer matters, so neither is waited for.
 */
const waitForMeasuredNodes = (flowElement: HTMLElement): Promise<boolean> => {
  // A removed node measures 0 x 0 too.
  const isPending = (node: HTMLElement) => {
    if (node.style.visibility !== 'hidden') {
      return false;
    }
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  let pending = Array.from(flowElement.querySelectorAll<HTMLElement>('.react-flow__node')).filter(isPending);
  if (!pending.length) {
    return Promise.resolve(true);
  }

  return new Promise((resolve) => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    const observer = new MutationObserver(() => {
      pending = pending.filter(isPending);
      if (!pending.length) {
        observer.disconnect();
        clearTimeout(timeoutId);
        resolve(true);
      }
    });
    observer.observe(flowElement, { attributeFilter: ['style'], attributes: true, childList: true, subtree: true });
    timeoutId = setTimeout(() => {
      observer.disconnect();
      resolve(false);
    }, WORKFLOW_EXPORT_LAYOUT_TIMEOUT_MS);
  });
};

const SOURCE_IMAGE_SELECTOR = '[data-workflow-export-field-value="true"] img';

/** Decode concurrently before html-to-image freezes computed image dimensions. */
const decodeSourceImages = async (flowElement: HTMLElement): Promise<Map<HTMLImageElement, string>> => {
  const images = Array.from(flowElement.querySelectorAll<HTMLImageElement>(SOURCE_IMAGE_SELECTOR));
  const decoded = new Map<HTMLImageElement, string>();
  if (!images.length) {
    return decoded;
  }
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      Promise.all(
        images.map((image) => {
          const source = image.src;
          return image.decode().then(
            () => {
              if (image.naturalWidth > 0 && image.naturalHeight > 0 && image.src === source) {
                decoded.set(image, source);
              }
            },
            () => undefined
          );
        })
      ),
      new Promise<void>((resolve) => {
        timeoutId = setTimeout(resolve, WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS);
      }),
    ]);
    // Late decodes must not change which images this export includes.
    return new Map(decoded);
  } finally {
    if (timeoutId !== undefined) {
      clearTimeout(timeoutId);
    }
  }
};

const replaceWithAltText = (image: HTMLImageElement) => {
  const fallback = document.createElement('span');
  fallback.textContent = image.alt;
  fallback.style.overflowWrap = 'anywhere';
  fallback.style.maxWidth = '100%';
  image.replaceWith(fallback);
};

const readAsDataUrl = (blob: Blob) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error('Workflow image source could not be read'));
    reader.readAsDataURL(blob);
  });

const embedImageBytes = async (image: HTMLImageElement, signal: AbortSignal) => {
  const response = await fetch(image.src, { signal });
  if (!response.ok) {
    throw new Error(`Workflow image source returned ${response.status}`);
  }
  image.src = await readAsDataUrl(await response.blob());
  await image.decode();
};

/**
 * Inlines decoded source images into the clone before html-to-image sees them, because its own embedding keeps every
 * fetched image in a module cache for the rest of the session. An image that did not decode, or whose bytes are not
 * fetched, read and decoded again within the image timeout, becomes its alt text so it can neither hang nor fail the
 * export; work still running past the timeout is abandoned.
 */
const embedSourceImages = async (
  flowElement: HTMLElement,
  clone: HTMLElement,
  decoded: Map<HTMLImageElement, string>
) => {
  const sourceImages = flowElement.querySelectorAll<HTMLImageElement>(SOURCE_IMAGE_SELECTOR);
  const controller = new AbortController();
  const timedOut = new Promise<never>((_, reject) => {
    controller.signal.addEventListener('abort', () => reject(new Error('Workflow image source timed out')), {
      once: true,
    });
  });
  // Observed by each image's race; this only keeps a timeout with no image left racing from going unhandled.
  timedOut.catch(() => undefined);
  const timeoutId = setTimeout(() => controller.abort(), WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS);
  try {
    await Promise.all(
      Array.from(clone.querySelectorAll<HTMLImageElement>(SOURCE_IMAGE_SELECTOR)).map(async (image, index) => {
        const original = sourceImages[index];
        if (!original || decoded.get(original) !== original.src || original.src !== image.src) {
          replaceWithAltText(image);
          return;
        }
        if (image.src.startsWith('data:')) {
          return;
        }
        try {
          await Promise.race([embedImageBytes(image, controller.signal), timedOut]);
        } catch {
          replaceWithAltText(image);
        }
      })
    );
  } finally {
    clearTimeout(timeoutId);
  }
};

/**
 * Refuses an export before any preparation when the rasterization slots are taken or the node bounds alone exceed the
 * budget. The caller can run this before mounting an export view; `exportWorkflowAsPng` repeats it.
 */
export const preflightWorkflowImageExport = (
  bounds: Rect,
  limits: WorkflowImageExportLimits = WORKFLOW_IMAGE_EXPORT_LIMITS
): WorkflowImageExportOutcome | null => {
  if (activeWorkflowRasterizations >= MAX_ACTIVE_WORKFLOW_RASTERIZATIONS) {
    return { status: 'busy' };
  }
  return planWorkflowImageExport(bounds, limits) ? null : { status: 'too-large' };
};

/**
 * Downloads the workflow as a PNG. Expected outcomes resolve; unexpected failures reject. The size budget is checked
 * from the node bounds before any waiting, and from measured content before cloning and again before capture.
 */
export const exportWorkflowAsPng = async ({
  flowElement,
  bounds,
  workflowName,
  fallbackWorkflowName,
  limits = WORKFLOW_IMAGE_EXPORT_LIMITS,
}: {
  flowElement: HTMLElement;
  bounds: Rect;
  workflowName: string;
  fallbackWorkflowName: string;
  limits?: WorkflowImageExportLimits;
}): Promise<WorkflowImageExportOutcome> => {
  const refusal = preflightWorkflowImageExport(bounds, limits);
  if (refusal) {
    return refusal;
  }

  const [measured, decoded] = await Promise.all([waitForMeasuredNodes(flowElement), decodeSourceImages(flowElement)]);
  if (!flowElement.isConnected) {
    return { status: 'canceled' };
  }
  if (!measured) {
    throw new Error('Workflow nodes were not measured before export');
  }
  const contentBounds = getWorkflowContentBounds(flowElement, bounds, { includeInputFieldLabels: false });
  const plan = planWorkflowImageExport(contentBounds, limits);
  if (!plan) {
    return { status: 'too-large' };
  }
  const clone = flowElement.cloneNode(true) as HTMLElement;
  const stagingWrapper = document.createElement('div');
  const captureRoot = document.createElement('div');

  try {
    await embedSourceImages(flowElement, clone, decoded);
    if (!flowElement.isConnected) {
      return { status: 'canceled' };
    }
    namespaceWorkflowExportIds(clone);
    prepareExportClone(clone, contentBounds, plan);
    Object.assign(stagingWrapper.style, getWorkflowExportStagingStyle(plan));
    captureRoot.appendChild(clone);
    stagingWrapper.appendChild(captureRoot);
    (flowElement.parentElement ?? document.body).appendChild(stagingWrapper);

    const measuredContentBounds = getWorkflowContentBounds(clone, contentBounds);
    const measuredPlan = planWorkflowImageExport(measuredContentBounds, limits);
    if (!measuredPlan) {
      return { status: 'too-large' };
    }
    if (
      measuredContentBounds.x !== contentBounds.x ||
      measuredContentBounds.y !== contentBounds.y ||
      measuredPlan.width !== plan.width ||
      measuredPlan.height !== plan.height
    ) {
      prepareExportClone(clone, measuredContentBounds, measuredPlan);
      Object.assign(stagingWrapper.style, getWorkflowExportStagingStyle(measuredPlan));
    }

    inlineSvgStylesForExport(clone);
    const backgroundColor = getComputedStyle(clone).backgroundColor;
    // Scaled only after measuring: the bounds above are read in unscaled layout pixels.
    const captureStyles = getWorkflowExportCaptureStyles(measuredPlan);
    Object.assign(captureRoot.style, captureStyles.root);
    Object.assign(clone.style, captureStyles.clone);
    if (activeWorkflowRasterizations >= MAX_ACTIVE_WORKFLOW_RASTERIZATIONS) {
      return { status: 'busy' };
    }
    const blob = await rasterizeWithTimeout(captureRoot, measuredPlan, backgroundColor);
    if (!flowElement.isConnected) {
      return { status: 'canceled' };
    }
    downloadPng(blob, workflowName, fallbackWorkflowName);
    return {
      status: 'exported',
      reduced: measuredPlan.scale < 1,
      width: measuredPlan.outputWidth,
      height: measuredPlan.outputHeight,
    };
  } finally {
    stagingWrapper.remove();
  }
};
