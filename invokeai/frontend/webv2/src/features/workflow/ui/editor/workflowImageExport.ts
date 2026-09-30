import type { Rect } from '@xyflow/react';

import { downloadBlob } from '@platform/browser/downloadBlob';

const WORKFLOW_GRID_SIZE = 25;

export const EXPORT_PADDING = 100;
export const EXPORT_SCALE = 2;
export const WORKFLOW_EXPORT_TIMEOUT_MS = 30_000;
export const WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS = 5_000;
const WORKFLOW_EXPORT_IMAGE_PLACEHOLDER = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
// html-to-image has no abort signal; bound captures that remain active after their caller times out.
const MAX_ACTIVE_WORKFLOW_RASTERIZATIONS = 2;
let activeWorkflowRasterizations = 0;
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

type WorkflowImageDimensions = {
  width: number;
  height: number;
  canvasWidth: number;
  canvasHeight: number;
};

const getPaddedWorkflowBounds = (bounds: Rect): Rect => ({
  x: bounds.x - EXPORT_PADDING,
  y: bounds.y - EXPORT_PADDING,
  width: bounds.width + EXPORT_PADDING * 2,
  height: bounds.height + EXPORT_PADDING * 2,
});

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

export const getWorkflowImageDimensions = (bounds: Rect): WorkflowImageDimensions => {
  const paddedBounds = getPaddedWorkflowBounds(bounds);

  const width = Math.max(1, Math.ceil(paddedBounds.width));
  const height = Math.max(1, Math.ceil(paddedBounds.height));

  const canvasWidth = width * EXPORT_SCALE;
  const canvasHeight = height * EXPORT_SCALE;

  return { width, height, canvasWidth, canvasHeight };
};

export const getWorkflowExportCloneStyle = (dimensions: WorkflowImageDimensions) => ({
  width: `${dimensions.width}px`,
  height: `${dimensions.height}px`,
  position: 'relative',
  left: '0',
  top: '0',
  pointerEvents: 'none',
});

export const getWorkflowExportOptions = (dimensions: WorkflowImageDimensions, backgroundColor: string) => ({
  width: dimensions.width,
  height: dimensions.height,
  canvasWidth: dimensions.canvasWidth,
  canvasHeight: dimensions.canvasHeight,
  backgroundColor,
  pixelRatio: 1,
  skipAutoScale: true,
  includeStyleProperties: [...EXPORT_STYLE_PROPERTIES],
  imagePlaceholder: WORKFLOW_EXPORT_IMAGE_PLACEHOLDER,
  onImageErrorHandler: () => WORKFLOW_EXPORT_IMAGE_PLACEHOLDER,
  skipFonts: false,
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

export const getWorkflowExportStagingStyle = (dimensions: WorkflowImageDimensions) => ({
  position: 'fixed',
  left: '-100000px',
  top: '0',
  width: `${dimensions.width}px`,
  height: `${dimensions.height}px`,
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

const prepareExportClone = (clone: HTMLElement, bounds: Rect, dimensions: WorkflowImageDimensions) => {
  const root = clone.matches('.react-flow') ? clone : clone.querySelector<HTMLElement>('.react-flow');
  const viewport = clone.querySelector<HTMLElement>('.react-flow__viewport');
  if (!root || !viewport) {
    throw new Error('Workflow editor DOM is missing React Flow viewport');
  }

  const translation = {
    x: EXPORT_PADDING - bounds.x,
    y: EXPORT_PADDING - bounds.y,
  };

  Object.assign(clone.style, getWorkflowExportCloneStyle(dimensions));

  root.style.width = `${dimensions.width}px`;
  root.style.height = `${dimensions.height}px`;
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

const toBlobWithTimeout = async (clone: HTMLElement, options: ReturnType<typeof getWorkflowExportOptions>) => {
  if (activeWorkflowRasterizations >= MAX_ACTIVE_WORKFLOW_RASTERIZATIONS) {
    throw new Error('A previous workflow image export is still running');
  }

  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  activeWorkflowRasterizations += 1;
  const releaseRasterizationSlot = () => {
    activeWorkflowRasterizations -= 1;
  };
  const rasterize = import('html-to-image').then(({ toBlob }) => (timedOut ? null : toBlob(clone, options)));
  void rasterize.then(releaseRasterizationSlot, releaseRasterizationSlot);

  try {
    return await Promise.race([
      rasterize,
      new Promise<never>((_, reject) => {
        timeoutId = setTimeout(() => {
          timedOut = true;
          reject(new Error(`Workflow image export timed out after ${WORKFLOW_EXPORT_TIMEOUT_MS} ms`));
        }, WORKFLOW_EXPORT_TIMEOUT_MS);
      }),
    ]);
  } finally {
    if (timeoutId !== undefined) {
      clearTimeout(timeoutId);
    }
  }
};

const downloadPng = (blob: Blob, workflowName: string, fallbackWorkflowName: string) => {
  downloadBlob(blob, `${sanitizeWorkflowImageFilename(workflowName, fallbackWorkflowName)}.png`);
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

const replaceFailedSourceImages = (
  flowElement: HTMLElement,
  clone: HTMLElement,
  decoded: Map<HTMLImageElement, string>
) => {
  const sourceImages = flowElement.querySelectorAll<HTMLImageElement>(SOURCE_IMAGE_SELECTOR);
  clone.querySelectorAll<HTMLImageElement>(SOURCE_IMAGE_SELECTOR).forEach((image, index) => {
    const original = sourceImages[index];
    if (original && decoded.get(original) === original.src && original.src === image.src) {
      return;
    }
    const fallback = document.createElement('span');
    fallback.textContent = image.alt;
    fallback.style.overflowWrap = 'anywhere';
    fallback.style.maxWidth = '100%';
    image.replaceWith(fallback);
  });
};

export const exportWorkflowAsPng = async ({
  flowElement,
  bounds,
  workflowName,
  fallbackWorkflowName,
}: {
  flowElement: HTMLElement;
  bounds: Rect;
  workflowName: string;
  fallbackWorkflowName: string;
}): Promise<void> => {
  if (activeWorkflowRasterizations >= MAX_ACTIVE_WORKFLOW_RASTERIZATIONS) {
    throw new Error('A previous workflow image export is still running');
  }

  const decoded = await decodeSourceImages(flowElement);
  if (!flowElement.isConnected) {
    throw new Error('Workflow image export canceled because the editor was unmounted');
  }
  const contentBounds = getWorkflowContentBounds(flowElement, bounds, { includeInputFieldLabels: false });
  const dimensions = getWorkflowImageDimensions(contentBounds);
  const clone = flowElement.cloneNode(true) as HTMLElement;
  const stagingWrapper = document.createElement('div');

  try {
    replaceFailedSourceImages(flowElement, clone, decoded);
    namespaceWorkflowExportIds(clone);
    prepareExportClone(clone, contentBounds, dimensions);
    Object.assign(stagingWrapper.style, getWorkflowExportStagingStyle(dimensions));
    stagingWrapper.appendChild(clone);
    (flowElement.parentElement ?? document.body).appendChild(stagingWrapper);

    const measuredContentBounds = getWorkflowContentBounds(clone, contentBounds);
    const measuredDimensions = getWorkflowImageDimensions(measuredContentBounds);
    if (
      measuredContentBounds.x !== contentBounds.x ||
      measuredContentBounds.y !== contentBounds.y ||
      measuredDimensions.width !== dimensions.width ||
      measuredDimensions.height !== dimensions.height
    ) {
      prepareExportClone(clone, measuredContentBounds, measuredDimensions);
      Object.assign(stagingWrapper.style, getWorkflowExportStagingStyle(measuredDimensions));
    }

    inlineSvgStylesForExport(clone);
    const blob = await toBlobWithTimeout(
      clone,
      getWorkflowExportOptions(measuredDimensions, getComputedStyle(clone).backgroundColor)
    );
    if (!blob) {
      throw new Error('Workflow image export returned an empty Blob');
    }
    if (!flowElement.isConnected) {
      throw new Error('Workflow image export canceled because the editor was unmounted');
    }
    downloadPng(blob, workflowName, fallbackWorkflowName);
  } finally {
    stagingWrapper.remove();
  }
};
