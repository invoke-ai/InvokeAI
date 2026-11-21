import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const uploads = vi.hoisted(() => ({
  blobs: [] as Blob[],
  options: [] as { isIntermediate?: boolean }[],
  onUpload: null as (() => Promise<void>) | null,
}));
vi.mock('./backend/canvasImages', () => ({
  uploadCanvasImage: async (blob: Blob, options: { isIntermediate?: boolean }) => {
    uploads.blobs.push(blob);
    uploads.options.push(options);
    await uploads.onUpload?.();
    const bitmap = await createImageBitmap(blob);
    try {
      return { imageName: `legacy-paint-${uploads.blobs.length}.png`, width: bitmap.width, height: bitmap.height };
    } finally {
      bitmap.close();
    }
  },
}));

import { convertLegacyCanvasMetadata } from './legacyCanvasMetadata';
import { prepareLegacyOpacityStroke } from './legacyCanvasPressure';

const pixel = async (blob: Blob, x: number, y: number): Promise<number[]> => {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  try {
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(bitmap, 0, 0);
    return [...ctx.getImageData(x, y, 1, 1).data];
  } finally {
    bitmap.close();
    canvas.width = 1;
    canvas.height = 1;
  }
};
const convert = (objects: unknown[], extra = {}, isCurrent?: () => boolean) =>
  convertLegacyCanvasMetadata(
    {
      canvas_v2_metadata: {
        rasterLayers: [
          {
            id: 'legacy',
            name: 'Painted',
            isEnabled: true,
            isLocked: false,
            position: { x: -50, y: 80 },
            opacity: 0.5,
            objects,
            ...extra,
          },
        ],
        controlLayers: [],
        inpaintMasks: [],
        regionalGuidance: [],
      },
    },
    captureAccountScope(),
    isCurrent
  );
const rectangle = {
  id: 'rect',
  type: 'rect',
  rect: { x: -8, y: 4, width: 16, height: 16 },
  color: { r: 200, g: 100, b: 50, a: 1 },
};

const pressureDot = {
  id: 'pressure',
  type: 'brush_line_with_pressure',
  points: [0, 0, 0.5],
  strokeWidth: 512,
  pressureAffectsOpacity: true,
  pressureAffectsWidth: false,
  color: { r: 255, g: 0, b: 0, a: 1 },
  clip: null,
};

beforeEach(() => {
  accountLifecycle.activate('legacy-canvas-browser');
  uploads.blobs.length = 0;
  uploads.options.length = 0;
  uploads.onUpload = null;
});

describe('legacy canvas raster recall', () => {
  it('bakes ordered objects and erasure with negative local bounds while preserving layer position and opacity', async () => {
    const result = await convert([
      rectangle,
      { id: 'erase', type: 'eraser_line', strokeWidth: 4, points: [0, 8, 0, 16], clip: null },
    ]);
    expect(result.problems).toEqual([]);
    const document = result.record!.document as {
      stacks: {
        raster: {
          transform: { x: number; y: number };
          opacity: number;
          source: { offset: { x: number; y: number } };
        }[];
      };
    };
    expect(document.stacks.raster[0]).toMatchObject({
      opacity: 0.5,
      transform: { x: -50, y: 80 },
      source: { offset: { x: -8, y: 4 } },
    });
    expect(await pixel(uploads.blobs[0]!, 2, 2)).toEqual([200, 100, 50, 255]);
    expect((await pixel(uploads.blobs[0]!, 8, 8))[3]).toBe(0);
  });

  it('bakes legacy adjustments and leaves alpha unchanged', async () => {
    const identity = [
      [0, 0],
      [255, 255],
    ];
    const result = await convert([rectangle], {
      adjustments: {
        version: 1,
        enabled: true,
        collapsed: false,
        mode: 'simple',
        simple: { brightness: 0.2, contrast: 0, saturation: 0, temperature: 0, tint: 0, sharpness: 0 },
        curves: { master: identity, r: identity, g: identity, b: identity },
      },
    });
    expect(result.problems).toEqual([]);
    expect(await pixel(uploads.blobs[0]!, 5, 5)).toEqual([251, 151, 101, 255]);
  });

  it('uses legacy maximum opacity coverage so overlapping pressure stamps do not darken the stroke', async () => {
    const result = await convert([
      {
        id: 'pressure',
        type: 'brush_line_with_pressure',
        points: [10, 10, 0.5, 20, 10, 0.5, 10, 10, 0.5],
        strokeWidth: 10,
        pressureAffectsOpacity: true,
        pressureAffectsWidth: false,
        color: { r: 255, g: 0, b: 0, a: 1 },
        clip: null,
      },
    ]);
    expect(result.problems).toEqual([]);
    const alpha = (await pixel(uploads.blobs[0]!, 10, 10))[3]!;
    expect(alpha).toBeGreaterThanOrEqual(126);
    expect(alpha).toBeLessThanOrEqual(128);
  });

  it.each([
    ['dot', [10, 10, 0.5]],
    ['horizontal stroke', [10, 10, 0.5, 20, 10, 0.5]],
  ] as const)('keeps a one-pixel pressure-opacity %s visible on its two-pixel-high output', async (_kind, points) => {
    // Legacy opacity is the maximum stamp coverage scaled by pressure, not repeated source-over blending.
    const reference = new OffscreenCanvas(5, 5);
    const ctx = reference.getContext('2d')!;
    ctx.beginPath();
    ctx.arc(2, 2, 0.5, 0, Math.PI * 2);
    ctx.fill();
    const expectedAlphas = [...ctx.getImageData(1, 1, 2, 2).data]
      .filter((_, index) => index % 4 === 3)
      .map((alpha) => Math.round((alpha * 128) / 255));
    reference.width = 1;
    reference.height = 1;
    expect(Math.min(...expectedAlphas)).toBeGreaterThan(0);
    const result = await convert([{ ...pressureDot, strokeWidth: 1, points: [...points] }]);
    expect(result.problems).toEqual([]);
    const bitmap = await createImageBitmap(uploads.blobs[0]!);
    expect(bitmap.height).toBe(2);
    bitmap.close();
    const last = points.length === 3 ? 1 : 11;
    for (let x = 0; x <= last; x++) {
      for (let y = 0; y < 2; y++) {
        const left = expectedAlphas[y * 2]!;
        const right = expectedAlphas[y * 2 + 1]!;
        const expected = x === 0 ? left : x === last ? right : Math.max(left, right);
        expect((await pixel(uploads.blobs[0]!, x, y))[3]).toBe(expected);
      }
    }
  });

  it.each(['overlapping dots', 'sparse full-layer buffers'] as const)(
    'preflights cumulative pressure work for %s before allocating or sampling a refused layer',
    async (kind) => {
      const objects =
        kind === 'overlapping dots'
          ? Array.from({ length: 10000 }, () => pressureDot)
          : [
              { ...rectangle, rect: { x: 0, y: 0, width: 1024, height: 1024 } },
              ...Array.from({ length: 100 }, () => ({ ...pressureDot, strokeWidth: 1, points: [512, 512, 0.5] })),
            ];
      const contexts = vi.spyOn(OffscreenCanvas.prototype, 'getContext');
      try {
        const result = await convert(objects);
        expect(result.record).toBeNull();
        expect(result.problems).toEqual(['Painted: legacy-conversion-work-limit']);
        expect(uploads.blobs).toHaveLength(0);
        expect(contexts).not.toHaveBeenCalled();
      } finally {
        contexts.mockRestore();
      }
    }
  );

  it('shares pressure work across layers and refuses the next layer before extra sampling or full-output buffers', async () => {
    const canvas = new OffscreenCanvas(1, 1);
    canvas.getContext('2d')!.fillRect(0, 0, 1, 1);
    const png = await canvas.convertToBlob({ type: 'image/png' });
    const encode = vi.spyOn(OffscreenCanvas.prototype, 'convertToBlob').mockResolvedValue(png);
    const reads = vi.spyOn(OffscreenCanvasRenderingContext2D.prototype, 'getImageData');
    const buffers = vi.spyOn(OffscreenCanvasRenderingContext2D.prototype, 'createImageData');
    try {
      const result = await convertLegacyCanvasMetadata(
        {
          canvas_v2_metadata: {
            rasterLayers: Array.from({ length: 11 }, (_, index) => ({
              id: `layer-${index}`,
              name: `Layer ${index}`,
              isEnabled: true,
              isLocked: false,
              position: { x: 0, y: 0 },
              opacity: 1,
              objects: [pressureDot],
            })),
            controlLayers: [],
            inpaintMasks: [],
            regionalGuidance: [],
          },
        },
        captureAccountScope()
      );
      expect(result.problems).toEqual(['Layer 0: legacy-conversion-work-limit']);
      expect((result.record!.document as { stacks: { raster: unknown[] } }).stacks.raster).toHaveLength(10);
      expect(uploads.blobs).toHaveLength(10);
      // Ten 516-square samples and ten 1024-square output buffers; the eleventh layer does neither.
      expect(reads.mock.calls.map((call) => [call[2], call[3]])).toEqual(Array.from({ length: 10 }, () => [516, 516]));
      expect(buffers).toHaveBeenCalledTimes(10);
      for (const call of buffers.mock.calls) {
        expect(call).toEqual([1024, 1024]);
      }
      expect(reads.mock.calls.reduce((sum, call) => sum + call[2]! * call[3]!, 0)).toBe(2_662_560);
    } finally {
      buffers.mockRestore();
      reads.mockRestore();
      encode.mockRestore();
      canvas.width = 1;
      canvas.height = 1;
    }
  });

  it.each([0, 4, 100])('refuses huge point inputs before smoothing with only %s remaining work units', (remaining) => {
    let pointReads = 0;
    const points = new Proxy(Array<number>(999_999).fill(0), {
      get(target, property, receiver) {
        if (typeof property === 'string' && /^\d+$/.test(property)) {
          pointReads++;
        }
        return Reflect.get(target, property, receiver);
      },
    });
    const budget = { workUnits: remaining };
    for (let attempt = 0; attempt < 100; attempt++) {
      expect(() => prepareLegacyOpacityStroke({ ...pressureDot, points }, { width: 1, height: 1 }, budget)).toThrow(
        'legacy-conversion-work-limit'
      );
    }
    expect(pointReads).toBe(0);
    expect(budget.workUnits).toBeGreaterThanOrEqual(0);
  });

  it('charges rejected stamp planning so later attempts cannot repeat raw-point smoothing or segment work', () => {
    let pointReads = 0;
    const points = new Proxy(Array.from({ length: 250 }, (_, index) => [index % 2 ? 10000 : 0, 0, 0.5]).flat(), {
      get(target, property, receiver) {
        if (typeof property === 'string' && /^\d+$/.test(property)) {
          pointReads++;
        }
        return Reflect.get(target, property, receiver);
      },
    });
    const planning = vi.spyOn(Math, 'hypot');
    const budget = { workUnits: 10000 };
    const prepare = () =>
      prepareLegacyOpacityStroke({ ...pressureDot, strokeWidth: 1, points }, { width: 1, height: 1 }, budget);
    try {
      expect(prepare).toThrow('legacy-conversion-work-limit');
      expect(pointReads).toBeGreaterThan(0);
      expect(planning).toHaveBeenCalledOnce();
      const firstPointReads = pointReads;
      for (let attempt = 0; attempt < 100; attempt++) {
        expect(prepare).toThrow('legacy-conversion-work-limit');
      }
      expect(pointReads).toBe(firstPointReads);
      expect(planning).toHaveBeenCalledOnce();
      expect(budget.workUnits).toBeGreaterThanOrEqual(0);
    } finally {
      planning.mockRestore();
    }
  });

  it('keeps rejected preparation charged across layers while retaining usable image-only layers', async () => {
    const points = Array.from({ length: 250 }, (_, index) => [index % 2 ? 10000 : 0, 0, 0.5]).flat();
    const layer = {
      id: 'legacy',
      name: 'Painted',
      isEnabled: true,
      isLocked: false,
      position: { x: 0, y: 0 },
      opacity: 1,
      objects: [{ ...pressureDot, points }],
    };
    const planning = vi.spyOn(Math, 'hypot');
    const contexts = vi.spyOn(OffscreenCanvas.prototype, 'getContext');
    try {
      const result = await convertLegacyCanvasMetadata(
        {
          canvas_v2_metadata: {
            rasterLayers: [
              {
                ...layer,
                id: 'saved',
                name: 'Already saved',
                objects: [
                  {
                    id: 'image',
                    type: 'image',
                    image: { image_name: 'saved.png', width: 64, height: 64 },
                  },
                ],
              },
              ...Array.from({ length: 100 }, (_, index) => ({ ...layer, id: `rejected-${index}` })),
            ],
            controlLayers: [],
            inpaintMasks: [],
            regionalGuidance: [],
          },
        },
        captureAccountScope()
      );
      expect(result.problems).toEqual(Array<string>(100).fill('Painted: legacy-conversion-work-limit'));
      expect((result.record!.document as { stacks: { raster: unknown[] } }).stacks.raster).toMatchObject([
        { id: 'saved', source: { type: 'image', image: { imageName: 'saved.png' } } },
      ]);
      // Only the first layer starts planning: its first segment exhausts the shared work reservation.
      expect(planning).toHaveBeenCalledOnce();
      expect(contexts).not.toHaveBeenCalled();
      expect(uploads.blobs).toHaveLength(0);
    } finally {
      contexts.mockRestore();
      planning.mockRestore();
    }
  });

  it('refuses oversized or malformed legacy objects without uploading a blank replacement', async () => {
    const result = await convert([{ ...rectangle, rect: { x: 0, y: 0, width: 1000000, height: 1000000 } }]);
    expect(result.record).toBeNull();
    expect(result.problems).toEqual(['Painted: legacy-layer-too-large']);
    expect(uploads.blobs).toHaveLength(0);
    expect((await convert([{ type: 'not-an-object' }])).record).toBeNull();
  });
});

it('stops a superseded project before raster uploads and bounds conversions in one untrusted record', async () => {
  const layer = {
    id: 'legacy',
    name: 'Painted',
    isEnabled: true,
    isLocked: false,
    position: { x: 0, y: 0 },
    opacity: 1,
    objects: [{ ...rectangle, rect: { x: 0, y: 0, width: 1, height: 1 } }],
  };
  const metadata = {
    canvas_v2_metadata: {
      rasterLayers: Array.from({ length: 129 }, (_, index) => ({
        ...layer,
        id: `layer-${index}`,
        name: `Layer ${index}`,
      })),
      controlLayers: [],
      inpaintMasks: [],
      regionalGuidance: [],
    },
  };
  await expect(convertLegacyCanvasMetadata(metadata, captureAccountScope(), () => false)).rejects.toThrow(
    'canvas-changed'
  );
  expect(uploads.blobs).toHaveLength(0);
  // Pixel tests above cover PNG encoding; this case exercises the per-record upload limit.
  const canvas = new OffscreenCanvas(1, 1);
  canvas.getContext('2d')!.fillRect(0, 0, 1, 1);
  const png = await canvas.convertToBlob({ type: 'image/png' });
  const encode = vi.spyOn(OffscreenCanvas.prototype, 'convertToBlob').mockResolvedValue(png);
  try {
    const result = await convertLegacyCanvasMetadata(metadata, captureAccountScope());
    expect(uploads.blobs).toHaveLength(128);
    expect(result.problems).toEqual(['Layer 0: legacy-conversion-limit']);
  } finally {
    encode.mockRestore();
  }
});

it.each(['project', 'account'] as const)(
  'leaves in-flight uploads reclaimable and refuses a %s change during the last upload',
  async (change) => {
    let markStarted!: () => void;
    let release!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let current = true;
    uploads.onUpload = () => {
      markStarted();
      return held;
    };
    const pending = convert([rectangle], {}, () => current);
    const settled = pending.catch(() => undefined);
    try {
      await Promise.race([
        started,
        settled.then(() => {
          throw new Error('Conversion ended before the upload started');
        }),
      ]);
      expect(uploads.blobs).toHaveLength(1);
      expect(uploads.options[0]).toMatchObject({ isIntermediate: true });
      if (change === 'project') {
        current = false;
      } else {
        accountLifecycle.activate('another-account');
      }
      const rejected = expect(pending).rejects.toThrow();
      release();
      await rejected;
      expect(uploads.blobs).toHaveLength(1);
    } finally {
      release();
      await settled;
    }
  }
);

it('keeps unanswered conversion uploads intermediate instead of leaving an unknown durable image', async () => {
  uploads.onUpload = () => Promise.reject(new TypeError('Failed to fetch'));
  const result = await convert([rectangle]);
  expect(result.record).toBeNull();
  expect(result.createdImageNames).toEqual([]);
  expect(uploads.options).toEqual([expect.objectContaining({ isIntermediate: true })]);
});

it('tracks only newly rasterized paint during a partial conversion, excluding borrowed image assets', async () => {
  const common = { name: 'Layer', isEnabled: true, isLocked: false, position: { x: 0, y: 0 }, opacity: 1 };
  const result = await convertLegacyCanvasMetadata(
    {
      canvas_v2_metadata: {
        rasterLayers: [
          { ...common, id: 'paint', objects: [rectangle] },
          {
            ...common,
            id: 'borrowed',
            objects: [{ id: 'image', type: 'image', image: { image_name: 'borrowed.png', width: 320, height: 200 } }],
          },
          { ...common, id: 'invalid', objects: [{ type: 'unsupported' }] },
        ],
        controlLayers: [],
        inpaintMasks: [],
        regionalGuidance: [],
      },
    },
    captureAccountScope()
  );
  expect(result.record).not.toBeNull();
  expect(result.problems).toHaveLength(1);
  expect(result.createdImageNames).toEqual(['legacy-paint-1.png']);
  expect(uploads.options).toEqual([expect.objectContaining({ isIntermediate: true })]);
});
