import type { ToolId } from '@workbench/canvas-engine/api';

import { describe, expect, it } from 'vitest';

import { OPERATION_PRESENTATION_ADAPTERS, TOOL_PRESENTATION_ADAPTERS } from './toolAdapters';

const TOOL_IDS = Object.keys({
  bbox: true,
  brush: true,
  colorPicker: true,
  eraser: true,
  gradient: true,
  lasso: true,
  marquee: true,
  move: true,
  sam: true,
  shape: true,
  text: true,
  transform: true,
  view: true,
} satisfies Record<ToolId, true>) as ToolId[];

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const en = Object.values(englishCatalogModules)[0] as {
  widgets: {
    canvas: {
      tools: Record<string, string>;
    };
    properties: { rows: Record<string, string>; sections: Record<string, string> };
  };
};

describe('tool presentation adapters', () => {
  it('registers every tool under its own id with a name in the English catalog', () => {
    for (const toolId of TOOL_IDS) {
      const adapter = TOOL_PRESENTATION_ADAPTERS[toolId];
      expect(adapter.id, toolId).toBe(toolId);
      expect(en.widgets.canvas.tools[toolId], toolId).toEqual(expect.any(String));
    }
  });

  it('gives every tool a form with at least one group', () => {
    for (const toolId of TOOL_IDS) {
      expect(TOOL_PRESENTATION_ADAPTERS[toolId].groups.length, toolId).toBeGreaterThan(0);
    }
  });

  it('resolves every form group label in the English catalog and keeps group ids unique per form', () => {
    const resolve = (key: string): unknown =>
      key.split('.').reduce<unknown>((node, part) => (node as Record<string, unknown> | undefined)?.[part], en);
    for (const toolId of TOOL_IDS) {
      const adapter = TOOL_PRESENTATION_ADAPTERS[toolId];
      const ids = adapter.groups.map((group) => group.id);
      expect(new Set(ids).size, toolId).toBe(ids.length);
      for (const group of adapter.groups) {
        expect(resolve(group.labelKey), `${toolId}:${group.id}`).toEqual(expect.any(String));
      }
    }
  });

  it('names the Properties sections in the catalog', () => {
    expect(en.widgets.properties.sections.tool).toEqual(expect.any(String));
    expect(en.widgets.properties.sections.operation).toEqual(expect.any(String));
  });

  it('gives every guarded operation groups and a footer so Apply and Cancel stay in place', () => {
    expect(Object.keys(OPERATION_PRESENTATION_ADAPTERS).sort()).toEqual(['filter', 'select-object']);
    for (const adapter of Object.values(OPERATION_PRESENTATION_ADAPTERS)) {
      expect(adapter.footer).toBeDefined();
      expect(adapter.groups.length).toBeGreaterThan(0);
    }
  });
});
