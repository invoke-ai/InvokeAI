import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  exportWorkflowAsPng,
  WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS,
  WORKFLOW_EXPORT_LAYOUT_TIMEOUT_MS,
} from './workflowImageExport';

let host: HTMLDivElement | null = null;
const objectUrls: string[] = [];

afterEach(() => {
  host?.remove();
  host = null;
  objectUrls.splice(0).forEach((url) => URL.revokeObjectURL(url));
  vi.restoreAllMocks();
  vi.useRealTimers();
});

const readPixels = async (image: Blob) => {
  const bitmap = await createImageBitmap(image);
  const { height, width } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d')!;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return { data: context.getImageData(0, 0, width, height).data, height, width };
};

/** Pixels that differ from the corner pixel, i.e. anything painted over the background. */
const countPaintedPixels = async (image: Blob): Promise<{ height: number; painted: number; width: number }> => {
  const { data, height, width } = await readPixels(image);
  const [r, g, b] = [data[0], data[1], data[2]];
  let painted = 0;

  for (let index = 0; index < data.length; index += 4) {
    if (Math.abs(data[index]! - r!) + Math.abs(data[index + 1]! - g!) + Math.abs(data[index + 2]! - b!) > 24) {
      painted += 1;
    }
  }

  return { height, painted, width };
};

const countColor = async (image: Blob, [r, g, b]: readonly [number, number, number]) => {
  const { data } = await readPixels(image);
  let count = 0;
  for (let index = 0; index < data.length; index += 4) {
    if (data[index] === r && data[index + 1] === g && data[index + 2] === b) {
      count += 1;
    }
  }
  return count;
};

const NODE_COLOR = [192, 36, 52] as const;

const solidImageUrl = async ([r, g, b]: readonly [number, number, number]) => {
  const canvas = new OffscreenCanvas(80, 40);
  const context = canvas.getContext('2d')!;
  context.fillStyle = `rgb(${r}, ${g}, ${b})`;
  context.fillRect(0, 0, 80, 40);
  const url = URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' }));
  objectUrls.push(url);
  return url;
};

const mountFlow = ({
  extraNodes = '',
  nodeContent = '',
  nodeVisibility = 'visible',
  nodeX = 20,
  nodeY = 20,
  zoom = 1,
}: {
  extraNodes?: string;
  nodeContent?: string;
  nodeVisibility?: 'hidden' | 'visible';
  nodeX?: number;
  nodeY?: number;
  zoom?: number;
} = {}) => {
  host = document.createElement('div');
  host.style.cssText = 'height:480px;position:relative;width:640px';
  document.body.append(host);
  const flowElement = document.createElement('div');
  flowElement.className = 'react-flow';
  flowElement.style.cssText = 'background-color:rgb(18, 20, 24);height:480px;position:relative;width:640px';
  flowElement.innerHTML = `
      <div class="react-flow__viewport" style="position:absolute;transform-origin:0 0;transform:translate(80px,-40px) scale(${zoom})">
        <div class="react-flow__edges" style="height:100%;position:absolute;width:100%">
          <svg height="120" width="200">
            <path class="react-flow__edge-path" d="M 80 45 C 110 45 120 45 150 45" style="fill:none;stroke:rgb(220, 220, 220);stroke-width:4px" />
          </svg>
        </div>
        <div class="react-flow__nodes" style="height:120px;position:absolute;width:200px">
          <div class="react-flow__node" style="height:60px;left:${nodeX}px;position:absolute;top:${nodeY}px;visibility:${nodeVisibility};width:90px">
            <div data-workflow-node-shell="true" style="background-color:rgb(${NODE_COLOR.join(', ')});height:60px;width:90px">Test node${nodeContent}</div>
          </div>${extraNodes}
        </div>
      </div>
      <div data-workflow-export-control="true" style="background-color:rgb(0, 255, 0);height:30px;position:absolute;width:30px"></div>
    `;
  host.append(flowElement);
  return flowElement;
};

const captureDownloads = () => {
  const clicked: { download: string }[] = [];
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function mockClick(this: HTMLAnchorElement) {
    clicked.push({ download: this.download });
  });
  const createObjectURL = vi.spyOn(URL, 'createObjectURL');
  const downloadedBlob = () => {
    const blob = createObjectURL.mock.calls.at(-1)?.[0];
    if (!(blob instanceof Blob)) {
      throw new Error('Workflow export did not download a Blob.');
    }
    return blob;
  };
  return { clicked, createObjectURL, downloadedBlob };
};

const exportOptions = (flowElement: HTMLElement) => ({
  bounds: { x: 0, y: 0, width: 200, height: 120 },
  fallbackWorkflowName: 'Unnamed Workflow',
  flowElement,
  workflowName: 'Camera workflow',
});

describe('workflow image export in the browser', () => {
  it.each([
    { zoom: 1, nodeX: 20, nodeY: 20, width: 660, height: 520 },
    { zoom: 0.5, nodeX: 20, nodeY: 20, width: 660, height: 520 },
    { zoom: 2, nodeX: -20, nodeY: -40, width: 740, height: 570 },
  ])('downloads a tight full resolution PNG at zoom $zoom', async ({ zoom, nodeX, nodeY, width, height }) => {
    const { clicked, createObjectURL, downloadedBlob } = captureDownloads();
    const flowElement = mountFlow({ nodeX, nodeY, zoom });

    await expect(exportWorkflowAsPng(exportOptions(flowElement))).resolves.toEqual({
      status: 'exported',
      reduced: false,
      width,
      height,
    });

    expect(clicked).toEqual([{ download: 'Camera workflow.png' }]);
    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(host!.children).toHaveLength(1);

    const blob = downloadedBlob();
    expect(blob.type).toBe('image/png');
    const { height: imageHeight, painted, width: imageWidth } = await countPaintedPixels(blob);
    expect([imageWidth, imageHeight]).toEqual([width, height]);
    expect(painted).toBeGreaterThan(width * height * 0.01);
    // The 90 x 60 node at 2x, less its text.
    expect(await countColor(blob, NODE_COLOR)).toBeGreaterThan(180 * 120 * 0.8);
  });

  it.each([
    // 330 x 260 layout pixels. 400 px on the long side is 400 / 330 = 1.21x: smaller than 2x, still above 1:1.
    { maxSide: 400, output: [400, 315], reduced: false, node: [109, 72] },
    // 300 / 330 = 0.91x is below the editor's 1:1 layout, which the outcome reports.
    { maxSide: 300, output: [300, 236], reduced: true, node: [81, 54] },
  ])(
    'exports a workflow over a $maxSide px side budget whole at a reduced scale',
    async ({ maxSide, node, output, reduced }) => {
      const { downloadedBlob } = captureDownloads();
      const flowElement = mountFlow();

      await expect(
        exportWorkflowAsPng({ ...exportOptions(flowElement), limits: { maxPixels: 1_000_000, maxSide, minScale: 0.5 } })
      ).resolves.toEqual({ status: 'exported', reduced, width: output[0], height: output[1] });

      const blob = downloadedBlob();
      const { height, width } = await countPaintedPixels(blob);
      expect([width, height]).toEqual(output);
      // The whole 90 x 60 node is still there, scaled rather than clipped, less its text.
      expect(await countColor(blob, NODE_COLOR)).toBeGreaterThan(node[0]! * node[1]! * 0.8);
    }
  );

  it('waits for React Flow to reveal a node it has not measured yet before capturing', async () => {
    const { createObjectURL, downloadedBlob } = captureDownloads();
    const flowElement = mountFlow({ nodeVisibility: 'hidden' });

    const exported = exportWorkflowAsPng(exportOptions(flowElement));
    // React Flow would have measured a mounted node within these frames; this one stays hidden until revealed.
    for (let frame = 0; frame < 2; frame += 1) {
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });
    }
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(host!.children).toHaveLength(1);

    flowElement.querySelector<HTMLElement>('.react-flow__node')!.style.visibility = 'visible';
    await expect(exported).resolves.toMatchObject({ status: 'exported' });

    expect(await countColor(downloadedBlob(), NODE_COLOR)).toBeGreaterThan(180 * 120 * 0.8);
  });

  it.each([
    { kind: 'removed', style: 'height:40px;width:40px' },
    { kind: 'zero-size', style: 'height:40px;width:0' },
  ])('does not wait on a hidden node that is $kind', async ({ kind, style }) => {
    // Only timeouts are faked: a wait that relied on the layout timeout would never finish.
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { downloadedBlob } = captureDownloads();
    const flowElement = mountFlow({
      extraNodes: `<div class="react-flow__node" data-extra-node style="left:150px;position:absolute;top:20px;visibility:hidden;${style}"></div>`,
    });

    const exported = exportWorkflowAsPng(exportOptions(flowElement));
    if (kind === 'removed') {
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });
      flowElement.querySelector('[data-extra-node]')!.remove();
    }
    await expect(exported).resolves.toMatchObject({ status: 'exported' });

    expect(await countColor(downloadedBlob(), NODE_COLOR)).toBeGreaterThan(180 * 120 * 0.8);
  });

  it('fails without capturing when nodes are never measured', async () => {
    vi.useFakeTimers();
    const { createObjectURL } = captureDownloads();
    const flowElement = mountFlow({ nodeVisibility: 'hidden' });

    const exported = exportWorkflowAsPng(exportOptions(flowElement));
    const rejection = expect(exported).rejects.toThrow('not measured');
    await vi.advanceTimersByTimeAsync(WORKFLOW_EXPORT_LAYOUT_TIMEOUT_MS);
    await rejection;

    expect(createObjectURL).not.toHaveBeenCalled();
    expect(host!.children).toHaveLength(1);
  });

  it('settles when the rendered page cannot be decoded, then exports again', async () => {
    const { createObjectURL } = captureDownloads();
    const flowElement = mountFlow();
    const decode = vi
      .spyOn(HTMLImageElement.prototype, 'decode')
      .mockRejectedValue(new DOMException('The source image cannot be decoded.', 'EncodingError'));

    await expect(exportWorkflowAsPng(exportOptions(flowElement))).rejects.toThrow('could not be decoded');
    expect(decode).toHaveBeenCalled();
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(host!.children).toHaveLength(1);

    decode.mockRestore();
    await expect(exportWorkflowAsPng(exportOptions(flowElement))).resolves.toMatchObject({ status: 'exported' });
    expect(createObjectURL).toHaveBeenCalledOnce();
  });

  it('rejects an empty PNG without downloading or leaving staging behind', async () => {
    const { createObjectURL } = captureDownloads();
    const flowElement = mountFlow();
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => callback(null));

    await expect(exportWorkflowAsPng(exportOptions(flowElement))).rejects.toThrow('empty Blob');
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(host!.children).toHaveLength(1);
  });

  it('substitutes alt text for a source image whose embedded copy never finishes decoding', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { downloadedBlob } = captureDownloads();
    const color = [40, 200, 90] as const;
    const flowElement = mountFlow({
      nodeContent: `<div data-workflow-export-field-value="true">
          <img alt="source.png" src="${await solidImageUrl(color)}" style="display:block;height:20px;width:40px">
        </div>`,
    });
    const decode = HTMLImageElement.prototype.decode;
    const embeddedDecodes: string[] = [];
    vi.spyOn(HTMLImageElement.prototype, 'decode').mockImplementation(function stalledDecode(this: HTMLImageElement) {
      if (this.src.startsWith('data:image/png')) {
        embeddedDecodes.push(this.src);
        return new Promise<void>(() => {});
      }
      return decode.call(this);
    });

    const exported = exportWorkflowAsPng(exportOptions(flowElement));
    await vi.waitFor(() => expect(embeddedDecodes).toHaveLength(1));
    await vi.advanceTimersByTimeAsync(WORKFLOW_EXPORT_IMAGE_TIMEOUT_MS);

    await expect(exported).resolves.toMatchObject({ status: 'exported' });
    expect(await countColor(downloadedBlob(), color)).toBe(0);
  });

  it('embeds source images per export and substitutes alt text once their bytes are gone', async () => {
    const { downloadedBlob } = captureDownloads();
    const color = [40, 200, 90] as const;
    const url = await solidImageUrl(color);
    const flowElement = mountFlow({
      nodeContent: `<div data-workflow-export-field-value="true">
          <img alt="source.png" src="${url}" style="display:block;height:20px;width:40px">
        </div>`,
    });
    const image = flowElement.querySelector('img')!;
    await image.decode();

    await expect(exportWorkflowAsPng(exportOptions(flowElement))).resolves.toMatchObject({ status: 'exported' });
    // 40 x 20 at 2x.
    expect(await countColor(downloadedBlob(), color)).toBeGreaterThan(80 * 40 * 0.8);

    // The editor keeps showing the decoded image, but nothing from the first export may stand in for its bytes.
    URL.revokeObjectURL(url);
    await expect(exportWorkflowAsPng(exportOptions(flowElement))).resolves.toMatchObject({ status: 'exported' });
    expect(await countColor(downloadedBlob(), color)).toBe(0);
    expect(flowElement.querySelector('img')).toBe(image);
  });
});
