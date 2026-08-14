import { describe, expect, it, vi } from 'vitest';

import {
  EXPORT_PADDING,
  EXPORT_SCALE,
  EXPORT_STYLE_PROPERTIES,
  getWorkflowExportCloneStyle,
  getWorkflowExportOptions,
  getWorkflowExportStagingStyle,
  getWorkflowImageDimensions,
  getWorkflowSvgExportStyles,
  hideWorkflowExportInfoIcons,
  hideWorkflowExportStatusIndicators,
  sanitizeWorkflowImageFilename,
  setWorkflowExportInputFieldTitleStyles,
  setWorkflowExportNodeOpacity,
  SVG_EXPORT_STYLE_PROPERTIES,
} from './workflowImageExport';

describe('workflow image export', () => {
  it('uses padded logical bounds and export scale for output dimensions', () => {
    expect(getWorkflowImageDimensions({ x: -100, y: 50, width: 1600, height: 900 })).toEqual({
      width: 1600 + EXPORT_PADDING * 2,
      height: 900 + EXPORT_PADDING * 2,
      canvasWidth: (1600 + EXPORT_PADDING * 2) * EXPORT_SCALE,
      canvasHeight: (900 + EXPORT_PADDING * 2) * EXPORT_SCALE,
    });
  });

  it('keeps capture clone local to an offscreen staging wrapper', () => {
    const dimensions = getWorkflowImageDimensions({ x: 0, y: 0, width: 400, height: 300 });

    expect(getWorkflowExportStagingStyle(dimensions)).toEqual({
      position: 'fixed',
      left: '-100000px',
      top: '0',
      width: '600px',
      height: '500px',
      pointerEvents: 'none',
    });
    expect(getWorkflowExportCloneStyle(dimensions)).toEqual({
      width: '600px',
      height: '500px',
      position: 'relative',
      left: '0',
      top: '0',
      pointerEvents: 'none',
    });
  });

  it('uses Blob-friendly image export dimensions', () => {
    const dimensions = getWorkflowImageDimensions({ x: 0, y: 0, width: 400, height: 300 });

    expect(getWorkflowExportOptions(dimensions, 'rgb(1, 2, 3)')).toEqual({
      width: 600,
      height: 500,
      canvasWidth: 1200,
      canvasHeight: 1000,
      backgroundColor: 'rgb(1, 2, 3)',
      pixelRatio: 1,
      includeStyleProperties: [...EXPORT_STYLE_PROPERTIES],
      skipFonts: true,
    });
  });

  it('preserves single-line field title styles in the export clone', () => {
    expect(EXPORT_STYLE_PROPERTIES).toEqual(
      expect.arrayContaining(['text-overflow', '-webkit-line-clamp', '-webkit-box-orient'])
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
    expect(setProperty).toHaveBeenCalledWith('overflow', 'hidden', 'important');
    expect(setProperty).toHaveBeenCalledWith('text-overflow', 'ellipsis', 'important');
  });

  it('keeps ordinary workflow names unchanged', () => {
    expect(sanitizeWorkflowImageFilename('Hi-Res Two Stage')).toBe('Hi-Res Two Stage');
  });

  it('replaces filesystem-invalid characters and falls back for blank names', () => {
    expect(sanitizeWorkflowImageFilename('Workflow: 01 / test?')).toBe('Workflow- 01 - test-');
    expect(sanitizeWorkflowImageFilename('   ...   ')).toBe('My Workflow');
  });
});
