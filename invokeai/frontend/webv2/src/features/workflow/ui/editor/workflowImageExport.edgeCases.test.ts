import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('html-to-image', () => ({
  toBlob: vi.fn(),
}));
vi.mock('@platform/browser/downloadBlob', () => ({
  downloadBlob: vi.fn(),
}));

import { downloadBlob } from '@platform/browser/downloadBlob';
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

  it('allows one bounded retry when a previous rasterization never settles', async () => {
    vi.useFakeTimers();
    let finishRasterization: (blob: Blob | null) => void = () => undefined;
    vi.mocked(toBlob).mockReturnValue(
      new Promise((resolve) => {
        finishRasterization = resolve;
      })
    );
    const { flowElement, stagingWrapper } = createExportDom();
    const exportOptions = {
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: 'Workflow',
      fallbackWorkflowName: 'Unnamed Workflow',
    };
    const exportPromise = exportWorkflowAsPng(exportOptions);
    const rejection = expect(exportPromise).rejects.toThrow('timed out');
    try {
      await vi.advanceTimersByTimeAsync(WORKFLOW_EXPORT_TIMEOUT_MS);

      await rejection;
      expect(stagingWrapper.remove).toHaveBeenCalledOnce();

      const retryBlob = new Blob(['png'], { type: 'image/png' });
      vi.mocked(toBlob).mockResolvedValueOnce(retryBlob);
      await exportWorkflowAsPng(exportOptions);
      expect(toBlob).toHaveBeenCalledTimes(2);
      expect(stagingWrapper.remove).toHaveBeenCalledTimes(2);
      finishRasterization(null);
      await vi.advanceTimersByTimeAsync(0);
    } finally {
      finishRasterization(null);
      await vi.advanceTimersByTimeAsync(0);
      vi.useRealTimers();
    }
  });

  it('downloads the rendered PNG using a sanitized workflow filename', async () => {
    const blob = new Blob(['png'], { type: 'image/png' });
    vi.mocked(toBlob).mockResolvedValue(blob);
    const { flowElement, stagingWrapper } = createExportDom();

    await exportWorkflowAsPng({
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: 'Workflow: 01 / test?',
      fallbackWorkflowName: 'Unnamed Workflow',
    });

    expect(downloadBlob).toHaveBeenCalledWith(blob, 'Workflow- 01 - test-.png');
    expect(stagingWrapper.remove).toHaveBeenCalledOnce();
  });

  it('uses the translated untitled name when downloading an unnamed workflow', async () => {
    const blob = new Blob(['png'], { type: 'image/png' });
    vi.mocked(toBlob).mockResolvedValue(blob);
    const { flowElement } = createExportDom();

    await exportWorkflowAsPng({
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: '',
      fallbackWorkflowName: 'Untitled Workflow',
    });

    expect(downloadBlob).toHaveBeenCalledWith(blob, 'Untitled Workflow.png');
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

  it('includes inline export descriptions in content bounds', () => {
    const description = {
      getBoundingClientRect: () => ({ left: 650, top: 250, width: 100, height: 50 }),
      scrollWidth: 100,
      scrollHeight: 50,
    };
    const viewport = { getBoundingClientRect: () => ({ left: 100, top: 200, width: 1000, height: 1000 }) };
    const flowElement = {
      getBoundingClientRect: () => ({ left: 100, top: 200 }),
      querySelector: (selector: string) => (selector === '.react-flow__viewport' ? viewport : null),
      querySelectorAll: (selector: string) =>
        selector === '[data-workflow-export-content="true"]' ? [description] : [],
    };

    expect(
      getWorkflowContentBounds(
        flowElement as unknown as HTMLElement,
        { x: 0, y: 0, width: 500, height: 100 },
        {
          includeInputFieldLabels: false,
        }
      )
    ).toMatchObject({ x: 0, y: 0, width: 650, height: 100 });
  });

  it('measures export content in logical coordinates when the workflow is zoomed out', () => {
    const description = {
      getBoundingClientRect: () => ({ left: 50, top: 50, width: 100, height: 25 }),
      scrollWidth: 200,
      scrollHeight: 50,
    };
    const viewport = { getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 1000 }) };
    const flowElement = {
      getBoundingClientRect: () => ({ left: 0, top: 0 }),
      querySelector: (selector: string) => (selector === '.react-flow__viewport' ? viewport : null),
      querySelectorAll: (selector: string) =>
        selector === '[data-workflow-export-content="true"]' ? [description] : [],
    };

    vi.stubGlobal('getComputedStyle', (element: unknown) => ({
      direction: 'ltr',
      transform: element === viewport ? 'matrix(0.5, 0, 0, 0.5, 0, 0)' : 'none',
    }));
    try {
      expect(
        getWorkflowContentBounds(
          flowElement as unknown as HTMLElement,
          { x: 0, y: 0, width: 100, height: 100 },
          { includeInputFieldLabels: false }
        )
      ).toEqual({ x: 0, y: 0, width: 300, height: 150 });
    } finally {
      vi.unstubAllGlobals();
    }
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
