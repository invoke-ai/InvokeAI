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
  WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS,
} from './workflowImageExport';

type FakeElement = {
  appendChild: (child: FakeElement) => void;
  attributes: Array<{ name: string; value: string }>;
  children: FakeElement[];
  cloneNode: () => FakeElement;
  id?: string;
  isConnected: boolean;
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
    isConnected: true,
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

  it.each(['failed', 'hung'] as const)(
    'replaces %s source images without changing the editor or retaining timers',
    async (state) => {
      vi.useFakeTimers();
      const { flowElement, clone, stagingWrapper } = createExportDom();
      const decode = vi.fn(() =>
        state === 'failed' ? Promise.reject(new Error('Unavailable')) : new Promise<void>(() => {})
      );
      const sourceImage = { decode, src: 'source.png' };
      const replacement = vi.fn();
      const clonedImage = { src: 'source.png', alt: 'source.png', replaceWith: replacement };
      const selector = '[data-workflow-export-field-value="true"] img';
      flowElement.querySelectorAll = (query) => (query === selector ? [sourceImage as unknown as FakeElement] : []);
      clone.querySelectorAll = (query) => (query === selector ? [clonedImage as unknown as FakeElement] : []);
      const fallback = { textContent: '', style: {} };
      vi.stubGlobal('document', {
        body: flowElement.parentElement,
        createElement: (tag: string) => (tag === 'span' ? fallback : stagingWrapper),
      });
      vi.mocked(toBlob).mockResolvedValue(new Blob(['png'], { type: 'image/png' }));
      const exportPromise = exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      });
      try {
        await vi.advanceTimersByTimeAsync(state === 'hung' ? WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS - 1 : 0);
        if (state === 'hung') {
          expect(toBlob).not.toHaveBeenCalled();
          await vi.advanceTimersByTimeAsync(1);
        }
        await exportPromise;
        expect(replacement).toHaveBeenCalledWith(fallback);
        expect(fallback.textContent).toBe('source.png');
        expect(flowElement.querySelectorAll(selector)).toEqual([sourceImage]);
        expect(downloadBlob).toHaveBeenCalledOnce();
        expect(stagingWrapper.remove).toHaveBeenCalledOnce();
        expect(vi.getTimerCount()).toBe(0);
      } finally {
        vi.useRealTimers();
      }
    }
  );

  it.each(['removed', 'replaced'] as const)(
    'matches decoded images after a source is %s during the wait',
    async (change) => {
      vi.useFakeTimers();
      const { flowElement, clone, stagingWrapper } = createExportDom();
      let failFirst!: () => void;
      const first = {
        src: 'first.png',
        decode: () =>
          new Promise<void>((_, reject) => {
            failFirst = () => reject(new Error('Unavailable'));
          }),
      };
      const second = { src: 'second.png', naturalWidth: 800, naturalHeight: 400, decode: () => Promise.resolve() };
      let images = [first, second];
      const replaceWith = vi.fn();
      const clonedImage = { src: 'second.png', alt: 'second.png', replaceWith };
      const selector = '[data-workflow-export-field-value="true"] img';
      flowElement.querySelectorAll = (query) => (query === selector ? (images as unknown as FakeElement[]) : []);
      clone.querySelectorAll = (query) => (query === selector ? [clonedImage as unknown as FakeElement] : []);
      const fallback = { textContent: '', style: {} };
      vi.stubGlobal('document', {
        body: flowElement.parentElement,
        createElement: (tag: string) => (tag === 'span' ? fallback : stagingWrapper),
      });
      vi.mocked(toBlob).mockResolvedValue(new Blob(['png'], { type: 'image/png' }));
      const exportPromise = exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      });
      try {
        await vi.advanceTimersByTimeAsync(0);
        images = [second];
        if (change === 'replaced') {
          second.src = 'new-source.png';
          clonedImage.src = 'new-source.png';
          clonedImage.alt = 'new-source.png';
        }
        failFirst();
        await exportPromise;
        if (change === 'removed') {
          expect(replaceWith).not.toHaveBeenCalled();
        } else {
          expect(replaceWith).toHaveBeenCalledWith(fallback);
          expect(fallback.textContent).toBe('new-source.png');
        }
        expect(vi.getTimerCount()).toBe(0);
      } finally {
        vi.useRealTimers();
      }
    }
  );

  it('cancels download if the editor unmounts during rasterization and removes staging', async () => {
    const { flowElement, stagingWrapper } = createExportDom();
    let finish!: (blob: Blob) => void;
    vi.mocked(toBlob).mockReturnValue(
      new Promise<Blob>((resolve) => {
        finish = resolve;
      })
    );
    const exportPromise = exportWorkflowAsPng({
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: 'Workflow',
      fallbackWorkflowName: 'Unnamed Workflow',
    });
    const rejection = expect(exportPromise).rejects.toThrow('canceled');
    await vi.waitFor(() => expect(toBlob).toHaveBeenCalledOnce());
    flowElement.isConnected = false;
    finish(new Blob(['png'], { type: 'image/png' }));
    await rejection;
    expect(downloadBlob).not.toHaveBeenCalled();
    expect(stagingWrapper.remove).toHaveBeenCalledOnce();
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

  it('includes overflowing output titles in content bounds', () => {
    const outputTitle = {
      getBoundingClientRect: () => ({ left: 650, top: 250, width: 100, height: 50 }),
      scrollWidth: 100,
      scrollHeight: 50,
    };
    const viewport = { getBoundingClientRect: () => ({ left: 100, top: 200, width: 1000, height: 1000 }) };
    const flowElement = {
      getBoundingClientRect: () => ({ left: 100, top: 200 }),
      querySelector: (selector: string) => (selector === '.react-flow__viewport' ? viewport : null),
      querySelectorAll: (selector: string) =>
        selector === '[data-workflow-export-output-title="true"]' ? [outputTitle] : [],
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

  it('includes full static field rows when measuring expanded snapshot content', () => {
    const fieldContent = {
      getBoundingClientRect: () => ({ left: 350, top: 120, width: 200, height: 420 }),
      scrollWidth: 200,
      scrollHeight: 420,
    };
    const viewport = { getBoundingClientRect: () => ({ left: 100, top: 100, width: 1000, height: 1000 }) };
    const flowElement = {
      getBoundingClientRect: () => ({ left: 100, top: 100 }),
      querySelector: (selector: string) => (selector === '.react-flow__viewport' ? viewport : null),
      querySelectorAll: (selector: string) =>
        selector === '[data-workflow-export-field-content="true"]' ? [fieldContent] : [],
    };

    vi.stubGlobal('getComputedStyle', () => ({ direction: 'ltr', transform: 'none' }));
    try {
      expect(
        getWorkflowContentBounds(
          flowElement as unknown as HTMLElement,
          { x: 0, y: 0, width: 300, height: 200 },
          { includeInputFieldLabels: false }
        )
      ).toMatchObject({ x: 0, y: 0, width: 450, height: 440 });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('includes expanded static node content in snapshot bounds', () => {
    const staticNodeContent = {
      getBoundingClientRect: () => ({ left: 350, top: 120, width: 200, height: 420 }),
      scrollWidth: 200,
      scrollHeight: 420,
    };
    const viewport = { getBoundingClientRect: () => ({ left: 100, top: 100, width: 1000, height: 1000 }) };
    const flowElement = {
      getBoundingClientRect: () => ({ left: 100, top: 100 }),
      querySelector: (selector: string) => (selector === '.react-flow__viewport' ? viewport : null),
      querySelectorAll: (selector: string) =>
        selector === '[data-workflow-export-static-node-content="true"]' ? [staticNodeContent] : [],
    };

    vi.stubGlobal('getComputedStyle', () => ({ direction: 'ltr', transform: 'none' }));
    try {
      expect(
        getWorkflowContentBounds(
          flowElement as unknown as HTMLElement,
          { x: 0, y: 0, width: 300, height: 200 },
          { includeInputFieldLabels: false }
        )
      ).toMatchObject({ x: 0, y: 0, width: 450, height: 440 });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('measures output titles in logical coordinates when the workflow is zoomed out', () => {
    const outputTitle = {
      getBoundingClientRect: () => ({ left: 50, top: 50, width: 100, height: 25 }),
      scrollWidth: 200,
      scrollHeight: 50,
    };
    const viewport = { getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 1000 }) };
    const flowElement = {
      getBoundingClientRect: () => ({ left: 0, top: 0 }),
      querySelector: (selector: string) => (selector === '.react-flow__viewport' ? viewport : null),
      querySelectorAll: (selector: string) =>
        selector === '[data-workflow-export-output-title="true"]' ? [outputTitle] : [],
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
