import { afterEach, describe, expect, it, vi } from 'vitest';

import { exportWorkflowAsPng } from './workflowImageExport';

let host: HTMLDivElement | null = null;

afterEach(() => {
  host?.remove();
  host = null;
  vi.restoreAllMocks();
});

/** Pixels that differ from the corner pixel, i.e. anything painted over the background. */
const countPaintedPixels = async (image: Blob): Promise<{ height: number; painted: number; width: number }> => {
  const bitmap = await createImageBitmap(image);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d')!;
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  const [r, g, b] = [data[0], data[1], data[2]];
  let painted = 0;

  for (let index = 0; index < data.length; index += 4) {
    if (Math.abs(data[index]! - r!) + Math.abs(data[index + 1]! - g!) + Math.abs(data[index + 2]! - b!) > 24) {
      painted += 1;
    }
  }

  return { height: bitmap.height, painted, width: bitmap.width };
};

describe('workflow image export in the browser', () => {
  it.each([
    { zoom: 1, nodeX: 20, nodeY: 20, width: 660, height: 520 },
    { zoom: 0.5, nodeX: 20, nodeY: 20, width: 660, height: 520 },
    { zoom: 2, nodeX: -20, nodeY: -40, width: 740, height: 570 },
    { zoom: 1, nodeX: 8500, nodeY: 20, width: 17420, height: 520, minimumPaintedPixels: 5000 },
  ])(
    'downloads a tight full resolution PNG at zoom $zoom',
    async ({ zoom, nodeX, nodeY, width, height, minimumPaintedPixels = width * height * 0.01 }) => {
      const clicked: { download: string }[] = [];
      vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function mockClick(this: HTMLAnchorElement) {
        clicked.push({ download: this.download });
      });
      const createObjectURL = vi.spyOn(URL, 'createObjectURL');

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
          <div class="react-flow__node" style="height:60px;left:${nodeX}px;position:absolute;top:${nodeY}px;width:90px">
            <div data-workflow-node-shell="true" style="background-color:rgb(192, 36, 52);height:60px;width:90px">Test node</div>
          </div>
        </div>
      </div>
      <div data-workflow-export-control="true" style="background-color:rgb(0, 255, 0);height:30px;position:absolute;width:30px"></div>
    `;
      host.append(flowElement);

      await exportWorkflowAsPng({
        bounds: { x: 0, y: 0, width: 200, height: 120 },
        fallbackWorkflowName: 'Unnamed Workflow',
        flowElement,
        workflowName: 'Camera workflow',
      });

      expect(clicked).toEqual([{ download: 'Camera workflow.png' }]);
      expect(createObjectURL).toHaveBeenCalledOnce();
      expect(host.children).toHaveLength(1);

      const capturedSource = createObjectURL.mock.calls[0]![0];
      if (!(capturedSource instanceof Blob)) {
        throw new Error('Workflow export did not download a Blob.');
      }

      const blob = capturedSource;
      expect(blob.type).toBe('image/png');
      const { height: imageHeight, painted, width: imageWidth } = await countPaintedPixels(blob);
      expect([imageWidth, imageHeight]).toEqual([width, height]);
      expect(painted).toBeGreaterThan(minimumPaintedPixels);
    }
  );
});
