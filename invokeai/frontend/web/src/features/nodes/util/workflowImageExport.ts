import type { Rect } from '@xyflow/react';
import { toBlob } from 'html-to-image';

export const EXPORT_PADDING = 100;
export const EXPORT_SCALE = 2;
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
  'line-height',
  'letter-spacing',
  'text-align',
  'text-decoration',
  'text-transform',
  'text-shadow',
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
const GRID_GAP = 25;
const DEFAULT_WORKFLOW_IMAGE_FILENAME = 'My Workflow';

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

const getWorkflowContentBounds = (flowElement: HTMLElement, nodeBounds: Rect): Rect => {
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

  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
};

export const getWorkflowImageDimensions = (bounds: Rect): WorkflowImageDimensions => {
  const paddedBounds = getPaddedWorkflowBounds(bounds);

  const width = Math.max(1, Math.ceil(paddedBounds.width));
  const height = Math.max(1, Math.ceil(paddedBounds.height));

  return { width, height, canvasWidth: width * EXPORT_SCALE, canvasHeight: height * EXPORT_SCALE };
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
  skipFonts: true,
});

export const getWorkflowSvgExportStyles = (computedStyle: Pick<CSSStyleDeclaration, 'getPropertyValue'>) =>
  SVG_EXPORT_STYLE_PROPERTIES.reduce<Record<string, string>>((styles, property) => {
    const value = computedStyle.getPropertyValue(property);
    if (value) {
      styles[property] = value;
    }
    return styles;
  }, {});

export const sanitizeWorkflowImageFilename = (workflowName: string): string => {
  const sanitizedName = workflowName
    .replace(/[<>:"/\\|?*]/g, '-')
    .split('')
    .map((character) => (character.charCodeAt(0) < 32 ? '-' : character))
    .join('')
    .trim()
    .replace(/[. ]+$/g, '');

  return sanitizedName || DEFAULT_WORKFLOW_IMAGE_FILENAME;
};

const setExportElementStyle = (element: HTMLElement, property: string, value: string) => {
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

  const patternId = `${pattern.id}-export`;
  pattern.id = patternId;
  pattern.setAttribute('width', `${GRID_GAP}`);
  pattern.setAttribute('height', `${GRID_GAP}`);
  pattern.setAttribute('x', `${((translation.x % GRID_GAP) + GRID_GAP) % GRID_GAP}`);
  pattern.setAttribute('y', `${((translation.y % GRID_GAP) + GRID_GAP) % GRID_GAP}`);
  pattern.setAttribute('patternTransform', `translate(-${GRID_GAP},-${GRID_GAP})`);

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

const downloadPng = (blob: Blob, workflowName: string) => {
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.download = `${sanitizeWorkflowImageFilename(workflowName)}.png`;
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
}: {
  flowElement: HTMLElement;
  bounds: Rect;
  workflowName: string;
}): Promise<void> => {
  const contentBounds = getWorkflowContentBounds(flowElement, bounds);
  const dimensions = getWorkflowImageDimensions(contentBounds);
  const clone = flowElement.cloneNode(true) as HTMLElement;
  const stagingWrapper = document.createElement('div');

  prepareExportClone(clone, contentBounds, dimensions);
  Object.assign(stagingWrapper.style, getWorkflowExportStagingStyle(dimensions));
  stagingWrapper.appendChild(clone);
  (flowElement.parentElement ?? document.body).appendChild(stagingWrapper);

  try {
    inlineSvgStylesForExport(clone);
    const blob = await toBlob(clone, getWorkflowExportOptions(dimensions, getComputedStyle(clone).backgroundColor));
    if (!blob) {
      throw new Error('Workflow image export returned an empty Blob');
    }
    downloadPng(blob, workflowName);
  } finally {
    stagingWrapper.remove();
  }
};
