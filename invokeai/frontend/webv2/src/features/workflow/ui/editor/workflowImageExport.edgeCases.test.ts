import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./workflowImageRaster', () => ({
  rasterizeWorkflowImage: vi.fn(),
}));
vi.mock('@platform/browser/downloadBlob', () => ({
  downloadBlob: vi.fn(),
}));

import { downloadBlob } from '@platform/browser/downloadBlob';

import {
  EXPORT_STYLE_PROPERTIES,
  exportWorkflowAsPng,
  getWorkflowContentBounds,
  WORKFLOW_EXPORT_TIMEOUT_MS,
  WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS,
} from './workflowImageExport';
import { rasterizeWorkflowImage } from './workflowImageRaster';

const rasterize = vi.mocked(rasterizeWorkflowImage);

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
  afterEach(() => {
    vi.unstubAllGlobals();
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
      rasterize.mockResolvedValue(new Blob(['png'], { type: 'image/png' }));
      const exportPromise = exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      });
      try {
        await vi.advanceTimersByTimeAsync(state === 'hung' ? WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS - 1 : 0);
        if (state === 'hung') {
          expect(rasterize).not.toHaveBeenCalled();
          await vi.advanceTimersByTimeAsync(1);
        }
        await expect(exportPromise).resolves.toMatchObject({ status: 'exported', reduced: false });
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
      const secondSource = 'data:image/png;base64,c2Vjb25k';
      const second = { src: secondSource, naturalWidth: 800, naturalHeight: 400, decode: () => Promise.resolve() };
      let images = [first, second];
      const replaceWith = vi.fn();
      const clonedImage = { src: secondSource, alt: 'second.png', replaceWith };
      const selector = '[data-workflow-export-field-value="true"] img';
      flowElement.querySelectorAll = (query) => (query === selector ? (images as unknown as FakeElement[]) : []);
      clone.querySelectorAll = (query) => (query === selector ? [clonedImage as unknown as FakeElement] : []);
      const fallback = { textContent: '', style: {} };
      vi.stubGlobal('document', {
        body: flowElement.parentElement,
        createElement: (tag: string) => (tag === 'span' ? fallback : stagingWrapper),
      });
      rasterize.mockResolvedValue(new Blob(['png'], { type: 'image/png' }));
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

  it('cancels download without failing if the editor unmounts during rasterization and removes staging', async () => {
    const { flowElement, stagingWrapper } = createExportDom();
    let finish!: (blob: Blob) => void;
    rasterize.mockReturnValue(
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
    await vi.waitFor(() => expect(rasterize).toHaveBeenCalledOnce());
    flowElement.isConnected = false;
    finish(new Blob(['png'], { type: 'image/png' }));
    await expect(exportPromise).resolves.toEqual({ status: 'canceled' });
    expect(downloadBlob).not.toHaveBeenCalled();
    expect(stagingWrapper.remove).toHaveBeenCalledOnce();
  });

  it('cancels before staging when the editor unmounts while source images are being embedded', async () => {
    const { flowElement, clone, stagingWrapper } = createExportDom();
    const selector = '[data-workflow-export-field-value="true"] img';
    const sourceImage = {
      decode: () => Promise.resolve(),
      naturalHeight: 40,
      naturalWidth: 80,
      src: 'https://invoke.test/source.png',
    };
    const clonedImage = { alt: 'source.png', replaceWith: vi.fn(), src: sourceImage.src };
    flowElement.querySelectorAll = (query) => (query === selector ? [sourceImage as unknown as FakeElement] : []);
    clone.querySelectorAll = (query) => (query === selector ? [clonedImage as unknown as FakeElement] : []);
    vi.stubGlobal('document', {
      body: flowElement.parentElement,
      createElement: (tag: string) => (tag === 'span' ? { style: {}, textContent: '' } : stagingWrapper),
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(() => {
        flowElement.isConnected = false;
        return Promise.reject(new TypeError('Failed to fetch'));
      })
    );

    await expect(
      exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      })
    ).resolves.toEqual({ status: 'canceled' });

    expect(clonedImage.replaceWith).toHaveBeenCalledOnce();
    expect(flowElement.parentElement!.children).toHaveLength(0);
    expect(rasterize).not.toHaveBeenCalled();
    expect(stagingWrapper.remove).toHaveBeenCalledOnce();
  });

  it('allows one bounded retry when a previous rasterization never settles', async () => {
    vi.useFakeTimers();
    let finishRasterization: (blob: Blob) => void = () => undefined;
    rasterize.mockReturnValue(
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
      // The timed-out capture is told to stop fetching and skip its remaining stages.
      expect(rasterize.mock.calls[0]![1].signal.aborted).toBe(true);

      const retryBlob = new Blob(['png'], { type: 'image/png' });
      rasterize.mockResolvedValueOnce(retryBlob);
      await exportWorkflowAsPng(exportOptions);
      expect(rasterize).toHaveBeenCalledTimes(2);
      expect(stagingWrapper.remove).toHaveBeenCalledTimes(2);
      finishRasterization(new Blob(['late'], { type: 'image/png' }));
      await vi.advanceTimersByTimeAsync(0);
      expect(downloadBlob).toHaveBeenCalledOnce();
    } finally {
      finishRasterization(new Blob(['late'], { type: 'image/png' }));
      await vi.advanceTimersByTimeAsync(0);
      vi.useRealTimers();
    }
  });

  it('refuses an oversized workflow from its node bounds before waiting on images or cloning', async () => {
    const { flowElement } = createExportDom();
    const cloneNode = vi.spyOn(flowElement, 'cloneNode');
    const decode = vi.fn(() => Promise.resolve());
    flowElement.querySelectorAll = (query) =>
      query === '[data-workflow-export-field-value="true"] img'
        ? [{ decode, src: 'a.png' } as unknown as FakeElement]
        : [];

    await expect(
      exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 40_000, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      })
    ).resolves.toEqual({ status: 'too-large' });

    expect(decode).not.toHaveBeenCalled();
    expect(cloneNode).not.toHaveBeenCalled();
    expect(rasterize).not.toHaveBeenCalled();
  });

  it('refuses before cloning when rendered content is larger than the node bounds', async () => {
    const { flowElement } = createExportDom();
    const cloneNode = vi.spyOn(flowElement, 'cloneNode');
    const node = createFakeElement({ getBoundingClientRect: () => ({ left: 0, top: 0, width: 40_000, height: 100 }) });
    flowElement.querySelectorAll = (query) => (query === '.react-flow__node' ? [node] : []);

    await expect(
      exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
      })
    ).resolves.toEqual({ status: 'too-large' });

    expect(cloneNode).not.toHaveBeenCalled();
    expect(rasterize).not.toHaveBeenCalled();
  });

  it('refuses before capture when the expanded clone outgrows the budget, and removes staging', async () => {
    const label = createFakeElement({
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 20 }),
      scrollWidth: 100,
      scrollHeight: 20,
    });
    label.style.setProperty = vi.fn((property: string) => {
      if (property === 'white-space') {
        label.scrollWidth = 5_000;
      }
    });
    const { clone, flowElement, stagingWrapper } = createExportDom();
    clone.querySelectorAll = (selector) => (selector === '[data-node-input-field-title="true"]' ? [label] : []);

    await expect(
      exportWorkflowAsPng({
        flowElement: flowElement as unknown as HTMLElement,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        workflowName: 'Workflow',
        fallbackWorkflowName: 'Unnamed Workflow',
        limits: { maxPixels: 1_000_000, maxSide: 1_000, minScale: 0.5 },
      })
    ).resolves.toEqual({ status: 'too-large' });

    expect(rasterize).not.toHaveBeenCalled();
    expect(stagingWrapper.remove).toHaveBeenCalledOnce();
  });

  it('downloads the rendered PNG using a sanitized workflow filename', async () => {
    const blob = new Blob(['png'], { type: 'image/png' });
    rasterize.mockResolvedValue(blob);
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
    rasterize.mockResolvedValue(blob);
    const { flowElement } = createExportDom();

    await exportWorkflowAsPng({
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: '',
      fallbackWorkflowName: 'Untitled Workflow',
    });

    expect(downloadBlob).toHaveBeenCalledWith(blob, 'Untitled Workflow.png');
  });

  it('includes overflowing input labels in content bounds', () => {
    vi.stubGlobal('getComputedStyle', () => ({ direction: 'ltr', transform: 'none' }));
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
    vi.stubGlobal('getComputedStyle', () => ({ direction: 'ltr', transform: 'none' }));
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
    rasterize.mockResolvedValue(new Blob(['png'], { type: 'image/png' }));
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
    ).resolves.toEqual({ status: 'exported', reduced: false, width: 1800, height: 640 });

    expect(rasterize.mock.calls[0]?.[1]).toMatchObject({ width: 1800, height: 640 });
  });

  it('preserves flex wrapping and document direction in the export clone', () => {
    expect(EXPORT_STYLE_PROPERTIES).toEqual(expect.arrayContaining(['flex-wrap', 'direction']));
  });

  it('does not put a duplicate workflow-editor id in the live document', async () => {
    rasterize.mockResolvedValue(new Blob(['png'], { type: 'image/png' }));
    const { clone, flowElement } = createExportDom();

    await exportWorkflowAsPng({
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: 'Workflow',
      fallbackWorkflowName: 'Unnamed Workflow',
    });

    expect(clone.id).not.toBe(flowElement.id);
  });

  it('namespaces cloned SVG ids and references', async () => {
    rasterize.mockResolvedValue(new Blob(['png'], { type: 'image/png' }));
    const marker = createFakeElement({ id: 'edge-marker' });
    const edgePath = createFakeElement({
      attributes: [{ name: 'marker-end', value: 'url(#edge-marker)' }],
    });
    const { clone, flowElement } = createExportDom();
    clone.querySelectorAll = (selector) => (selector === '*' ? [marker, edgePath] : []);

    await exportWorkflowAsPng({
      flowElement: flowElement as unknown as HTMLElement,
      bounds: { x: 0, y: 0, width: 100, height: 100 },
      workflowName: 'Workflow',
      fallbackWorkflowName: 'Unnamed Workflow',
    });

    expect(marker.id).toBe('edge-marker-workflow-export');
    expect(edgePath.attributes).toEqual([{ name: 'marker-end', value: 'url(#edge-marker-workflow-export)' }]);
  });
});
