import { describe, expect, it, vi } from 'vitest';

import {
  EXPORT_STYLE_PROPERTIES,
  getWorkflowExportCloneStyle,
  getWorkflowExportStagingStyle,
  getWorkflowSvgExportStyles,
  hideWorkflowExportInfoIcons,
  hideWorkflowExportStatusIndicators,
  planWorkflowImageExport,
  sanitizeWorkflowImageFilename,
  setWorkflowExportInputFieldTitleStyles,
  setWorkflowExportNodeOpacity,
  SVG_EXPORT_STYLE_PROPERTIES,
} from './workflowImageExport';

describe('workflow image export', () => {
  it('exports ordinary workflows at full resolution from padded layout bounds', () => {
    expect(planWorkflowImageExport({ x: -100, y: 50, width: 1600, height: 900 })).toEqual({
      width: 1800,
      height: 1100,
      scale: 2,
      outputWidth: 3600,
      outputHeight: 2200,
    });
  });

  it.each([
    // The long side is capped at 16,384 px and the short side keeps the aspect ratio: floor(500 * 16384 / 20200).
    { kind: 'wide', bounds: [20000, 300], output: [16384, 405] },
    { kind: 'tall', bounds: [300, 20000], output: [405, 16384] },
    // 9,200 px square would be 18,400 px square (338,560,000 px) at 2x; floor(sqrt(2^25)) = 5,792 px square fits.
    { kind: 'large', bounds: [9000, 9000], output: [5792, 5792] },
  ])('reduces resolution for a very $kind workflow to stay within the default budget', ({ bounds, output }) => {
    const plan = planWorkflowImageExport({ x: 0, y: 0, width: bounds[0]!, height: bounds[1]! });

    expect(plan).not.toBeNull();
    expect(plan!.scale).toBeLessThan(2);
    expect([plan!.outputWidth, plan!.outputHeight]).toEqual(output);
    expect(plan!.outputWidth * plan!.outputHeight).toBeLessThanOrEqual(2 ** 25);
  });

  it.each([
    { kind: 'area', bounds: [12000, 12000] },
    { kind: 'side', bounds: [40000, 100] },
  ])('refuses a workflow whose $kind exceeds the budget even at half scale', ({ bounds }) => {
    expect(planWorkflowImageExport({ x: 0, y: 0, width: bounds[0]!, height: bounds[1]! })).toBeNull();
  });

  const SIDE_BOUND = { maxPixels: 1_000_000, maxSide: 1000, minScale: 0.5 };
  // Sides up to 2,000 px never bind for these rows; only the 500,000 px area does.
  const AREA_BOUND = { maxPixels: 500_000, maxSide: 2000, minScale: 0.5 };

  it.each([
    { edge: 'side at the limit', limits: SIDE_BOUND, bounds: [300, 100], output: [1000, 600], reduced: false },
    { edge: 'side one pixel over', limits: SIDE_BOUND, bounds: [301, 100], output: [1000, 598], reduced: true },
    // 500 x 250 layout px is exactly 1,000 x 500 = 500,000 px at 2x.
    { edge: 'area at the limit', limits: AREA_BOUND, bounds: [300, 50], output: [1000, 500], reduced: false },
    // 500 x 251 needs sqrt(500,000 / 125,500) = 1.996x: floor(998.0) x floor(500.99).
    { edge: 'area one row over', limits: AREA_BOUND, bounds: [300, 51], output: [998, 500], reduced: true },
    { edge: 'exactly the minimum scale', limits: SIDE_BOUND, bounds: [1800, 0], output: [1000, 100], reduced: true },
  ])('plans the $edge boundary within injected limits', ({ bounds, limits, output, reduced }) => {
    const plan = planWorkflowImageExport({ x: 0, y: 0, width: bounds[0]!, height: bounds[1]! }, limits);

    expect([plan?.outputWidth, plan?.outputHeight]).toEqual(output);
    expect(plan!.scale < 2).toBe(reduced);
  });

  it('refuses just below the minimum scale', () => {
    expect(
      planWorkflowImageExport(
        { x: 0, y: 0, width: 1801, height: 0 },
        { maxPixels: 1_000_000, maxSide: 1000, minScale: 0.5 }
      )
    ).toBeNull();
  });

  it('rejects non-finite bounds instead of planning a capture', () => {
    expect(() => planWorkflowImageExport({ x: 0, y: 0, width: Number.NaN, height: 100 })).toThrow('not finite');
    expect(() => planWorkflowImageExport({ x: 0, y: 0, width: 100, height: Infinity })).toThrow('not finite');
  });

  it('keeps capture clone local to an offscreen staging wrapper', () => {
    const plan = planWorkflowImageExport({ x: 0, y: 0, width: 400, height: 300 })!;

    expect(getWorkflowExportStagingStyle(plan)).toEqual({
      position: 'fixed',
      left: '-100000px',
      top: '0',
      width: '600px',
      height: '500px',
      pointerEvents: 'none',
    });
    expect(getWorkflowExportCloneStyle(plan)).toEqual({
      width: '600px',
      height: '500px',
      position: 'relative',
      left: '0',
      top: '0',
      pointerEvents: 'none',
    });
  });

  it('preserves single-line field title styles in the export clone', () => {
    expect(EXPORT_STYLE_PROPERTIES).toEqual(
      expect.arrayContaining(['aspect-ratio', 'text-overflow', '-webkit-line-clamp', '-webkit-box-orient'])
    );
  });

  it('extracts computed SVG edge styles for inline capture', () => {
    const computedStyle = {
      getPropertyValue: (property: string) =>
        ({ stroke: 'rgb(1, 2, 3)', 'stroke-width': '3px', fill: 'none' })[property] ?? '',
    };

    expect(getWorkflowSvgExportStyles(computedStyle)).toEqual({
      stroke: 'rgb(1, 2, 3)',
      'stroke-width': '3px',
      fill: 'none',
    });
    expect(SVG_EXPORT_STYLE_PROPERTIES).toContain('stroke');
    expect(SVG_EXPORT_STYLE_PROPERTIES).toContain('marker-end');
  });

  it('makes node wrappers opaque regardless of the node opacity slider', () => {
    const setProperty = vi.fn();
    const nodeWrapper = { style: { setProperty } } as unknown as HTMLElement;
    const root = {
      querySelectorAll: (selector: string) =>
        selector === '.react-flow__node > [data-is-selected]' ? [nodeWrapper] : [],
    } as unknown as HTMLElement;

    setWorkflowExportNodeOpacity(root);

    expect(setProperty).toHaveBeenCalledWith('opacity', '1', 'important');
  });

  it('hides node status indicators from the export clone', () => {
    const setProperty = vi.fn();
    const statusIndicator = { style: { setProperty } } as unknown as HTMLElement;
    const root = {
      querySelectorAll: (selector: string) =>
        selector === '[data-node-status-indicator="true"]' ? [statusIndicator] : [],
    } as unknown as HTMLElement;

    hideWorkflowExportStatusIndicators(root);

    expect(setProperty).toHaveBeenCalledWith('display', 'none', 'important');
  });

  it('hides node information icons from the export clone', () => {
    const setProperty = vi.fn();
    const infoIcon = { style: { setProperty } } as unknown as SVGElement;
    const root = {
      querySelectorAll: (selector: string) => (selector === '[data-node-info-icon="true"]' ? [infoIcon] : []),
    } as unknown as HTMLElement;

    hideWorkflowExportInfoIcons(root);

    expect(setProperty).toHaveBeenCalledWith('display', 'none', 'important');
  });

  it('keeps input field titles on one line in the export clone', () => {
    const setProperty = vi.fn();
    const fieldTitle = { style: { setProperty } } as unknown as HTMLElement;
    const root = {
      querySelectorAll: (selector: string) => (selector === '[data-node-input-field-title="true"]' ? [fieldTitle] : []),
    } as unknown as HTMLElement;

    setWorkflowExportInputFieldTitleStyles(root);

    expect(setProperty).toHaveBeenCalledWith('display', 'block', 'important');
    expect(setProperty).toHaveBeenCalledWith('white-space', 'nowrap', 'important');
    expect(setProperty).toHaveBeenCalledWith('overflow', 'visible', 'important');
    expect(setProperty).toHaveBeenCalledWith('text-overflow', 'clip', 'important');
  });

  it('keeps ordinary workflow names unchanged', () => {
    expect(sanitizeWorkflowImageFilename('Hi-Res Two Stage', 'Unnamed Workflow')).toBe('Hi-Res Two Stage');
  });

  it('replaces filesystem-invalid characters and falls back for blank names', () => {
    expect(sanitizeWorkflowImageFilename('Workflow: 01 / test?', 'Unnamed Workflow')).toBe('Workflow- 01 - test-');
    expect(sanitizeWorkflowImageFilename('   ...   ', 'Unnamed Workflow')).toBe('Unnamed Workflow');
  });
});
