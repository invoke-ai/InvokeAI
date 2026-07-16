import { describe, expect, it } from 'vitest';

import { resolveCanvasOptionsContent, TOOL_OPTIONS_COMPONENTS } from './ToolOptionsBar';

describe('TOOL_OPTIONS_COMPONENTS', () => {
  it('has an entry for exactly the tools with dedicated options today (bbox, brush, eraser, gradient, lasso, move, shape, text, transform)', () => {
    expect(Object.keys(TOOL_OPTIONS_COMPONENTS).sort()).toEqual([
      'bbox',
      'brush',
      'eraser',
      'gradient',
      'lasso',
      'move',
      'shape',
      'text',
      'transform',
    ]);
  });

  it('has no entry for the view tool — the bar shows only the doc info for it', () => {
    expect(TOOL_OPTIONS_COMPONENTS.view).toBeUndefined();
  });

  it('has no entry for tools not yet implemented by the engine', () => {
    for (const toolId of ['colorPicker', 'sam'] as const) {
      expect(TOOL_OPTIONS_COMPONENTS[toolId]).toBeUndefined();
    }
  });

  it('every registered entry is a defined component function', () => {
    const components = Object.values(TOOL_OPTIONS_COMPONENTS);
    expect(components.length).toBeGreaterThan(0);
    for (const component of components) {
      expect(typeof component).toBe('function');
    }
  });

  it('gives an active canvas operation priority over the active or temporary tool', () => {
    expect(resolveCanvasOptionsContent({ status: 'active' }, 'sam')).toBe('operation');
    expect(resolveCanvasOptionsContent({ status: 'active' }, 'view')).toBe('operation');
    expect(resolveCanvasOptionsContent({ status: 'idle' }, 'brush')).toBe('brush');
    expect(resolveCanvasOptionsContent({ status: 'idle' }, 'view')).toBeNull();
  });
});
