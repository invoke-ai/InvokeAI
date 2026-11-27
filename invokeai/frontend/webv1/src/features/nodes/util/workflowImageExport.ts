import type { Rect } from '@xyflow/react';
import { WORKFLOW_GRID_SIZE } from 'features/nodes/types/constants';
import { toBlob } from 'html-to-image';

export const EXPORT_PADDING = 100;
export const EXPORT_SCALE = 2;
export const EXPORT_MAX_CANVAS_DIMENSION = 16_384;
export const WORKFLOW_EXPORT_TIMEOUT_MS = 30_000;
const WORKFLOW_EXPORT_IMAGE_PLACEHOLDER = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
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
  let minX = nodeBounds.x;
  let minY = nodeBounds.y;
  let maxX = nodeBounds.x + nodeBounds.width;
  let maxY = nodeBounds.y + nodeBounds.height;

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

  if (includeInputFieldLabels) {
    const flowRect = flowElement.getBoundingClientRect();
    const viewport = flowElement.querySelector<HTMLElement>('.react-flow__viewport');
    const viewportRect = viewport?.getBoundingClientRect() ?? flowRect;
    const transform = viewport ? (getComputedStyle(viewport).transform ?? 'none') : 'none';
    const matrix = transform
      .match(/^matrix\(([^)]+)\)$/)?.[1]
      ?.split(',')
      .map(Number);
    const zoom = matrix?.[0] && Number.isFinite(matrix[0]) && matrix[0] > 0 ? matrix[0] : 1;

    flowElement.querySelectorAll<HTMLElement>('[data-node-input-field-title="true"]').forEach((label) => {
      const labelRect = label.getBoundingClientRect();
      const intrinsicWidth = Math.max(labelRect.width, label.scrollWidth);
      const intrinsicHeight = Math.max(labelRect.height, label.scrollHeight);
      const overflowWidth = Math.max(0, intrinsicWidth - labelRect.width) / zoom;
      const direction = getComputedStyle(label).direction;
      const labelX = (labelRect.left - viewportRect.left) / zoom - (direction === 'rtl' ? overflowWidth : 0);
      const labelY = (labelRect.top - viewportRect.top) / zoom;
      const labelWidth = intrinsicWidth / zoom;
      const labelHeight = intrinsicHeight / zoom;

      minX = Math.min(minX, labelX);
      minY = Math.min(minY, labelY);
      maxX = Math.max(maxX, labelX + labelWidth);
      maxY = Math.max(maxY, labelY + labelHeight);
    });
  }

  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
};

export const getWorkflowImageDimensions = (bounds: Rect): WorkflowImageDimensions => {
  const paddedBounds = getPaddedWorkflowBounds(bounds);

  const width = Math.max(1, Math.ceil(paddedBounds.width));
  const height = Math.max(1, Math.ceil(paddedBounds.height));

  const scale = Math.min(EXPORT_SCALE, EXPORT_MAX_CANVAS_DIMENSION / width, EXPORT_MAX_CANVAS_DIMENSION / height);
  const canvasWidth = Math.max(1, Math.floor(width * scale));
  const canvasHeight = Math.max(1, Math.floor(height * scale));

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
  setExportElementStyle(root, 'background-color', 'var(--invoke-colors-base-900)');
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
  clone.querySelectorAll<HTMLElement>('[data-connector-node-body="true"]').forEach((element) => {
    setExportElementStyle(element, 'background-color', 'var(--invoke-colors-base-700)');
  });
  clone.querySelectorAll<HTMLElement>('[data-connector-node-icon="true"]').forEach((element) => {
    setExportElementStyle(element, 'color', 'var(--invoke-colors-base-100)');
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
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  try {
    return await Promise.race([
      toBlob(clone, options),
      new Promise<never>((_, reject) => {
        timeoutId = setTimeout(() => {
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
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.download = `${sanitizeWorkflowImageFilename(workflowName, fallbackWorkflowName)}.png`;
  anchor.href = objectUrl;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
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
  const contentBounds = getWorkflowContentBounds(flowElement, bounds, { includeInputFieldLabels: false });
  const dimensions = getWorkflowImageDimensions(contentBounds);
  const clone = flowElement.cloneNode(true) as HTMLElement;
  const stagingWrapper = document.createElement('div');

  try {
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
    downloadPng(blob, workflowName, fallbackWorkflowName);
  } finally {
    stagingWrapper.remove();
  }
};
