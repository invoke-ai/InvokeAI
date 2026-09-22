import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('html-to-image', () => ({
  toBlob: vi.fn(),
}));

import { toBlob } from 'html-to-image';

import {
  EXPORT_STYLE_PROPERTIES,
  exportWorkflowAsPng,
  getWorkflowContentBounds,
  getWorkflowExportOptions,
  WORKFLOW_EXPORT_TIMEOUT_MS,
} from './workflowImageExport';

type FakeElement = {
  appendChild: (child: FakeElement) => void;
  attributes: Array<{ name: string; value: string }>;
  children: FakeElement[];
  cloneNode: () => FakeElement;
  id?: string;
  matches: () => boolean;
  parentElement: FakeElement | null;
  getBoundingClientRect: () => { left: number; top: number; width: number; height: number };
  querySelector: (selector: string) => FakeElement | null;
  querySelectorAll: (selector: string) => FakeElement[];
  remove: () => void;
  setAttribute: (name: string, value: string) => void;
  scrollHeight?: number;
  scrollWidth?: number;
  style: { setProperty: ReturnType<typeof vi.fn> } & Record<string, unknown>;
};

const createFakeElement = (overrides: Partial<FakeElement> = {}): FakeElement => {
  const element: FakeElement = {
    appendChild: (child) => element.children.push(child),
    attributes: [],
    children: [],
    cloneNode: () => element,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 1000 }),
    matches: () => false,
    parentElement: null,
    querySelector: () => null,
    querySelectorAll: () => [],
    remove: vi.fn(),
    setAttribute: (name, value) => {
      const attribute = element.attributes.find((candidate) => candidate.name === name);
      if (attribute) {
        attribute.value = value;
      } else {
        element.attributes.push({ name, value });
      }
    },
    style: { setProperty: vi.fn() },
    ...overrides,
  };
  return element;
};

const createExportDom = () => {
  const parent = createFakeElement();
  const root = createFakeElement();
  const viewport = createFakeElement();
  const clone = createFakeElement({ id: 'workflow-editor' });
  const stagingWrapper = createFakeElement();
  const flowElement = createFakeElement({ id: 'workflow-editor', parentElement: parent, cloneNode: () => clone });
  flowElement.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1000, height: 1000 });

  clone.querySelector = (selector) => {
    if (selector === '.react-flow') {
      return root;
    }
    if (selector === '.react-flow__viewport') {
      return viewport;
    }
    return null;
  };
  stagingWrapper.remove = vi.fn();

  vi.stubGlobal('document', {
    body: parent,
    createElement: () => stagingWrapper,
  });
  vi.stubGlobal('getComputedStyle', () => ({ backgroundColor: 'rgb(1, 2, 3)', transform: 'none', direction: 'ltr' }));

  return { clone, flowElement, stagingWrapper };
};

describe('workflow image export edge cases', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('settles and cleans up when rasterization never settles', async () => {
    vi.useFakeTimers();
    vi.mocked(toBlob).mockReturnValue(new Promise<Blob | null>(() => {}));
    const { flowElement, stagingWrapper } = createExportDom();
    const exportPromise = exportWorkflowAsPng({
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: 'Workflow',
      fallbackWorkflowName: 'Unnamed Workflow',
    });
    const rejection = expect(exportPromise).rejects.toThrow('timed out');
    try {
      await vi.advanceTimersByTimeAsync(WORKFLOW_EXPORT_TIMEOUT_MS);

      await rejection;
      expect(stagingWrapper.remove).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
    }
  });

  it('configures failed image embedding to degrade instead of aborting export', () => {
    const options = getWorkflowExportOptions(
      { width: 100, height: 100, canvasWidth: 200, canvasHeight: 200 },
      'rgb(1, 2, 3)'
    );

    expect(options.imagePlaceholder).toBeTruthy();
  });

  it('keeps the Invoke font available to the serialized image', () => {
    const options = getWorkflowExportOptions(
      { width: 100, height: 100, canvasWidth: 200, canvasHeight: 200 },
      'rgb(1, 2, 3)'
    );

    expect(options.skipFonts).toBe(false);
  });

  it('includes overflowing input labels in content bounds', () => {
    const label = {
      getBoundingClientRect: () => ({ left: 590, top: 220, width: 100, height: 20 }),
      scrollWidth: 200,
      scrollHeight: 20,
    };
    const viewport = { getBoundingClientRect: () => ({ left: 100, top: 200, width: 1000, height: 1000 }) };
    const flowElement = {
      getBoundingClientRect: () => ({ left: 100, top: 200 }),
      querySelector: (selector: string) => (selector === '.react-flow__viewport' ? viewport : null),
      querySelectorAll: (selector: string) => (selector === '[data-node-input-field-title="true"]' ? [label] : []),
    };

    expect(
      getWorkflowContentBounds(flowElement as unknown as HTMLElement, { x: 0, y: 0, width: 500, height: 100 })
    ).toMatchObject({ x: 0, y: 0, width: 690, height: 100 });
  });

  it('measures overflowing labels after export styles are applied', async () => {
    vi.mocked(toBlob).mockResolvedValue(null);
    const label = createFakeElement({
      getBoundingClientRect: () => ({ left: 500, top: 100, width: 100, height: 20 }),
      scrollWidth: 100,
      scrollHeight: 20,
    });
    label.style.setProperty = vi.fn((property: string) => {
      if (property === 'white-space') {
        label.scrollWidth = 200;
      }
    });
    const { clone, flowElement } = createExportDom();
    clone.querySelectorAll = (selector) => (selector === '[data-node-input-field-title="true"]' ? [label] : []);

    await expect(
      exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      })
    ).rejects.toThrow('empty Blob');

    expect(vi.mocked(toBlob).mock.calls[0]?.[1]).toMatchObject({ width: 900, height: 320 });
  });

  it('preserves flex wrapping and document direction in the export clone', () => {
    expect(EXPORT_STYLE_PROPERTIES).toEqual(expect.arrayContaining(['flex-wrap', 'direction']));
  });

  it('does not put a duplicate workflow-editor id in the live document', async () => {
    vi.mocked(toBlob).mockResolvedValue(null);
    const { clone, flowElement } = createExportDom();

    await expect(
      exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      })
    ).rejects.toThrow('empty Blob');

    expect(clone.id).not.toBe(flowElement.id);
  });

  it('namespaces cloned SVG ids and references', async () => {
    vi.mocked(toBlob).mockResolvedValue(null);
    const marker = createFakeElement({ id: 'edge-marker' });
    const edgePath = createFakeElement({
      attributes: [{ name: 'marker-end', value: 'url(#edge-marker)' }],
    });
    const { clone, flowElement } = createExportDom();
    clone.querySelectorAll = (selector) => (selector === '*' ? [marker, edgePath] : []);

    await expect(
      exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      })
    ).rejects.toThrow('empty Blob');

    expect(marker.id).toBe('edge-marker-workflow-export');
    expect(edgePath.attributes).toEqual([{ name: 'marker-end', value: 'url(#edge-marker-workflow-export)' }]);
  });
});
