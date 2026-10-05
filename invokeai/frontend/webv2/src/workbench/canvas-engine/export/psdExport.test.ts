import type { CanvasBlendMode } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { readPsd, writePsd } from 'ag-psd';
import { describe, expect, it } from 'vitest';

import type {
  PsdExportGroupInput,
  PsdExportLayerInput,
  PsdExportNodeInput,
  PsdPlanFolder,
  PsdPlanNode,
} from './psdExport';

import { blendModeToPsd, executePsdExport, planPsdExport, PSD_MAX_DIMENSION } from './psdExport';

const IDENTITY = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

const layer = (over: Partial<PsdExportLayerInput> = {}): PsdExportLayerInput => ({
  blendMode: 'normal',
  contentRect: { height: 50, width: 100, x: 0, y: 0 },
  id: 'a',
  isEnabled: true,
  name: 'Layer',
  opacity: 1,
  transform: { ...IDENTITY },
  ...over,
});

const findLayer = (plan: ReturnType<typeof planPsdExport>, id: string) => {
  if (plan.status !== 'ok') {
    throw new Error(`expected ok plan, got ${plan.status}`);
  }
  const found = plan.layers.find((l) => l.id === id);
  if (!found) {
    throw new Error(`layer ${id} not in plan`);
  }
  return found;
};

describe('blendModeToPsd', () => {
  it('maps every canvas blend mode to a PSD blend key', () => {
    const cases: [CanvasBlendMode, string][] = [
      ['normal', 'normal'],
      ['multiply', 'multiply'],
      ['screen', 'screen'],
      ['overlay', 'overlay'],
      ['darken', 'darken'],
      ['lighten', 'lighten'],
      ['color-dodge', 'color dodge'],
      ['color-burn', 'color burn'],
      ['hard-light', 'hard light'],
      ['soft-light', 'soft light'],
      ['difference', 'difference'],
      ['exclusion', 'exclusion'],
      ['hue', 'hue'],
      ['saturation', 'saturation'],
      ['color', 'color'],
      ['luminosity', 'luminosity'],
    ];
    for (const [mode, key] of cases) {
      expect(blendModeToPsd(mode)).toBe(key);
    }
  });

  it('falls back to normal for an unknown blend mode', () => {
    expect(blendModeToPsd('made-up' as CanvasBlendMode)).toBe('normal');
  });
});

describe('planPsdExport', () => {
  it('returns empty when there are no layers', () => {
    expect(planPsdExport([])).toEqual({ status: 'empty' });
  });

  it('returns empty when every layer has empty content', () => {
    expect(planPsdExport([layer({ contentRect: { height: 0, width: 0, x: 0, y: 0 } })])).toEqual({
      status: 'empty',
    });
  });

  it('sizes the PSD canvas to a single layer content bounds', () => {
    const plan = planPsdExport([layer()]);
    expect(plan.status).toBe('ok');
    if (plan.status !== 'ok') {
      return;
    }
    expect(plan.width).toBe(100);
    expect(plan.height).toBe(50);
    expect(plan.canvasRect).toEqual<Rect>({ height: 50, width: 100, x: 0, y: 0 });
    const only = plan.layers[0]!;
    expect(only).toMatchObject({ bottom: 50, hidden: false, left: 0, opacity: 1, right: 100, top: 0 });
  });

  it('unions world-space bounds across layers and positions each relative to the origin', () => {
    const plan = planPsdExport([
      layer({ id: 'top', transform: { ...IDENTITY, x: 50, y: 20 } }),
      layer({ id: 'bottom', transform: { ...IDENTITY, x: -30, y: -10 } }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    // union of [-30,-10,100,50] and [50,20,100,50] = [-30,-10, 180, 80]
    expect(plan.canvasRect).toEqual<Rect>({ height: 80, width: 180, x: -30, y: -10 });
    expect(plan.width).toBe(180);
    expect(plan.height).toBe(80);
    // positions are relative to the union origin (-30, -10)
    expect(findLayer(plan, 'bottom')).toMatchObject({ left: 0, top: 0, right: 100, bottom: 50 });
    expect(findLayer(plan, 'top')).toMatchObject({ left: 80, top: 30, right: 180, bottom: 80 });
  });

  it('emits layers bottom-to-top (input is top-first, PSD children are bottom-first)', () => {
    const plan = planPsdExport([layer({ id: 'top' }), layer({ id: 'mid' }), layer({ id: 'bottom' })]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    expect(plan.layers.map((l) => l.id)).toEqual(['bottom', 'mid', 'top']);
  });

  it('marks disabled layers hidden but still exports them, and clamps opacity to 0..1', () => {
    const plan = planPsdExport([
      layer({ id: 'shown', isEnabled: true, opacity: 0.5 }),
      layer({ id: 'over', isEnabled: true, opacity: 2 }),
      layer({ id: 'hidden', isEnabled: false, opacity: -1 }),
    ]);
    expect(findLayer(plan, 'shown')).toMatchObject({ hidden: false, opacity: 0.5 });
    expect(findLayer(plan, 'over')).toMatchObject({ opacity: 1 });
    expect(findLayer(plan, 'hidden')).toMatchObject({ hidden: true, opacity: 0 });
  });

  it('maps blend modes and reports unmapped ones (falling back to normal)', () => {
    const plan = planPsdExport([
      layer({ blendMode: 'multiply', id: 'a' }),
      layer({ blendMode: 'nonsense' as CanvasBlendMode, id: 'b' }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    expect(findLayer(plan, 'a').blendMode).toBe('multiply');
    expect(findLayer(plan, 'b').blendMode).toBe('normal');
    expect(plan.unmappedBlends).toEqual(['nonsense']);
  });

  it('passes non-destructive adjustments through for the executor to bake', () => {
    const adjustments = [
      { brightness: 0.2, contrast: 0, id: 'adj-bc', isEnabled: true, type: 'brightness-contrast' as const },
    ];
    const plan = planPsdExport([layer({ adjustments })]);
    expect(findLayer(plan, 'a').adjustments).toBe(adjustments);
  });

  it('drops empty-content layers from the plan but keeps the rest', () => {
    const plan = planPsdExport([
      layer({ contentRect: { height: 0, width: 0, x: 0, y: 0 }, id: 'empty' }),
      layer({ id: 'real' }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    expect(plan.layers.map((l) => l.id)).toEqual(['real']);
  });

  it('refuses an export whose union bounds exceed the dimension cap', () => {
    const plan = planPsdExport([layer({ contentRect: { height: 10, width: 100, x: 0, y: 0 } })], {
      maxDimension: 50,
    });
    expect(plan).toEqual({ height: 10, status: 'too-large', width: 100 });
  });

  it('accepts bounds exactly at the cap', () => {
    const plan = planPsdExport([layer({ contentRect: { height: 10, width: PSD_MAX_DIMENSION, x: 0, y: 0 } })]);
    expect(plan.status).toBe('ok');
  });
});

describe('planPsdExport — folders', () => {
  const group = (id: string, children: PsdExportNodeInput[], isEnabled = true): PsdExportGroupInput => ({
    children,
    id,
    isEnabled,
    name: id,
    type: 'group',
  });
  const names = (nodes: readonly PsdPlanNode[]): unknown[] =>
    nodes.map((node) => (node.kind === 'folder' ? { [node.name]: names(node.children) } : node.name));

  it('mirrors the tree as folders, reversing every level, and flattens leaves in visual order', () => {
    const plan = planPsdExport([
      layer({ id: 'top', name: 'Top' }),
      group('G', [
        layer({ id: 'g1', name: 'G1' }),
        group('H', [layer({ id: 'h1', name: 'H1' })]),
        layer({ id: 'g2', name: 'G2' }),
      ]),
      layer({ id: 'bottom', name: 'Bottom' }),
    ]);
    expect(plan.status).toBe('ok');
    if (plan.status !== 'ok') {
      return;
    }
    expect(names(plan.tree)).toEqual(['Bottom', { G: ['G2', { H: ['H1'] }, 'G1'] }, 'Top']);
    expect(plan.layers.map((entry) => entry.id)).toEqual(['bottom', 'g2', 'h1', 'g1', 'top']);
  });

  it('writes own visibility on folders and leaves while the preview flattens only contributing leaves', () => {
    const plan = planPsdExport([
      group(
        'off',
        [layer({ id: 'inner', name: 'Inner' }), layer({ id: 'dark', isEnabled: false, name: 'Dark' })],
        false
      ),
      layer({ id: 'root', name: 'Root' }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error(plan.status);
    }
    const folder = plan.tree[1] as PsdPlanFolder;
    expect(folder).toMatchObject({ hidden: true, kind: 'folder', name: 'off' });
    expect(findLayer(plan, 'inner')).toMatchObject({ contributes: false, hidden: false });
    expect(findLayer(plan, 'dark')).toMatchObject({ contributes: false, hidden: true });
    expect(findLayer(plan, 'root')).toMatchObject({ contributes: true, hidden: false });
  });

  it('drops a folder whose subtree exports nothing', () => {
    const empty = { height: 0, width: 0, x: 0, y: 0 };
    const plan = planPsdExport([
      group('hollow', [group('deeper', [layer({ contentRect: empty, id: 'nothing' })])]),
      layer({ id: 'root', name: 'Root' }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error(plan.status);
    }
    expect(names(plan.tree)).toEqual(['Root']);
  });

  it('plans folder opacity and blend (defaulting when absent) and reports unmapped folder blends', () => {
    const plan = planPsdExport([
      group('plain', [layer({ id: 'p1' })]),
      { ...group('styled', [layer({ id: 's1' })]), blendMode: 'multiply', opacity: 0.25 },
      { ...group('odd', [layer({ id: 'o1' })]), blendMode: 'plasma' as CanvasBlendMode },
    ]);
    if (plan.status !== 'ok') {
      throw new Error(plan.status);
    }
    const folders = plan.tree.filter((node): node is PsdPlanFolder => node.kind === 'folder');
    // A default group is pass-through; writing 'normal' would isolate it in
    // Photoshop and change how members blend with layers below the group.
    expect(folders.find((f) => f.id === 'plain')).toMatchObject({
      blendMode: 'pass through',
      compositeBlend: 'source-over',
      opacity: 1,
    });
    expect(folders.find((f) => f.id === 'styled')).toMatchObject({
      blendMode: 'multiply',
      compositeBlend: 'multiply',
      opacity: 0.25,
    });
    // An unmapped blend still isolates (it is non-normal); only the KEY falls back.
    expect(folders.find((f) => f.id === 'odd')).toMatchObject({ blendMode: 'normal' });
    expect(plan.unmappedBlends).toEqual(['plasma']);
  });

  it('round-trips folder opacity and blend through ag-psd', async () => {
    const plan = planPsdExport([
      { ...group('Styled', [layer({ id: 'in', name: 'In' })]), blendMode: 'screen', opacity: 0.5 },
      layer({ id: 'base', name: 'Base' }),
    ]);
    const backend = createTestStubRasterBackend();
    const imageDataOf = (width: number, height: number): ImageData =>
      ({ data: new Uint8ClampedArray(width * height * 4), height, width }) as ImageData;
    let bytes: ArrayBuffer | null = null;
    await executePsdExport(plan, 'styled.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: () =>
        Promise.resolve({ rect: { height: 50, width: 100, x: 0, y: 0 }, surface: backend.createSurface(100, 50) }),
      readImageData: (_surface, rect) => imageDataOf(rect.width, rect.height),
      writeImageData: () => undefined,
      writePsd: (psd) => Promise.resolve(writePsd(psd, { generateThumbnail: false })),
    });
    const parsed = readPsd(bytes!, { skipCompositeImageData: true, skipLayerImageData: true, skipThumbnail: true });
    const folder = parsed.children!.find((child) => child.name === 'Styled')!;
    expect(folder.blendMode).toBe('screen');
    expect(folder.opacity).toBeCloseTo(0.5, 2);
  });

  it('round-trips color labels as native PSD layer colors on layers and folders', async () => {
    const plan = planPsdExport([
      { ...group('Tagged', [layer({ colorLabel: 'red', id: 'in', name: 'In' })]), colorLabel: 'violet' },
      layer({ id: 'base', name: 'Base' }),
    ]);
    const backend = createTestStubRasterBackend();
    const imageDataOf = (width: number, height: number): ImageData =>
      ({ data: new Uint8ClampedArray(width * height * 4), height, width }) as ImageData;
    let bytes: ArrayBuffer | null = null;
    await executePsdExport(plan, 'labels.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: () =>
        Promise.resolve({ rect: { height: 50, width: 100, x: 0, y: 0 }, surface: backend.createSurface(100, 50) }),
      readImageData: (_surface, rect) => imageDataOf(rect.width, rect.height),
      writeImageData: () => undefined,
      writePsd: (psd) => Promise.resolve(writePsd(psd, { generateThumbnail: false })),
    });
    const parsed = readPsd(bytes!, { skipCompositeImageData: true, skipLayerImageData: true, skipThumbnail: true });
    const folder = parsed.children!.find((child) => child.name === 'Tagged')!;
    expect(folder.layerColor).toBe('violet');
    expect(folder.children![0]!.layerColor).toBe('red');
    expect(parsed.children!.find((child) => child.name === 'Base')!.layerColor ?? 'none').toBe('none');
  });

  it('round-trips a default folder as pass-through', async () => {
    const plan = planPsdExport([
      group('Plain', [layer({ id: 'in', name: 'In' })]),
      layer({ id: 'base', name: 'Base' }),
    ]);
    const backend = createTestStubRasterBackend();
    const imageDataOf = (width: number, height: number): ImageData =>
      ({ data: new Uint8ClampedArray(width * height * 4), height, width }) as ImageData;
    let bytes: ArrayBuffer | null = null;
    await executePsdExport(plan, 'plain.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: () =>
        Promise.resolve({ rect: { height: 50, width: 100, x: 0, y: 0 }, surface: backend.createSurface(100, 50) }),
      readImageData: (_surface, rect) => imageDataOf(rect.width, rect.height),
      writeImageData: () => undefined,
      writePsd: (psd) => Promise.resolve(writePsd(psd, { generateThumbnail: false })),
    });
    const parsed = readPsd(bytes!, { skipCompositeImageData: true, skipLayerImageData: true, skipThumbnail: true });
    expect(parsed.children!.find((child) => child.name === 'Plain')!.blendMode).toBe('pass through');
  });

  it('round-trips folder hierarchy, order, names and leaf properties through ag-psd', async () => {
    const plan = planPsdExport([
      group('Shading', [layer({ blendMode: 'multiply', id: 'shade', name: 'Shade', opacity: 0.5 })], false),
      layer({ id: 'base', name: 'Base' }),
    ]);
    const backend = createTestStubRasterBackend();
    const imageDataOf = (width: number, height: number): ImageData =>
      ({ data: new Uint8ClampedArray(width * height * 4), height, width }) as ImageData;
    let bytes: ArrayBuffer | null = null;
    await executePsdExport(plan, 'tree.psd', {
      backend,
      download: (data) => {
        bytes = data;
      },
      getLayerSurface: () =>
        Promise.resolve({ rect: { height: 50, width: 100, x: 0, y: 0 }, surface: backend.createSurface(100, 50) }),
      readImageData: (_surface, rect) => imageDataOf(rect.width, rect.height),
      writeImageData: () => undefined,
      writePsd: (psd) => Promise.resolve(writePsd(psd, { generateThumbnail: false })),
    });
    const parsed = readPsd(bytes!, { skipCompositeImageData: true, skipLayerImageData: true, skipThumbnail: true });
    expect(parsed.children?.map((child) => child.name)).toEqual(['Base', 'Shading']);
    const folder = parsed.children![1]!;
    expect(folder.hidden).toBe(true);
    expect(folder.children?.map((child) => child.name)).toEqual(['Shade']);
    expect(folder.children![0]).toMatchObject({ blendMode: 'multiply', hidden: false });
    // Opacity rides an 8-bit field in the file.
    expect(folder.children![0]!.opacity).toBeCloseTo(0.5, 2);
  });
});
