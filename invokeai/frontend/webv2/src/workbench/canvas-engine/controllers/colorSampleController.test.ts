import type { ColorSamplePhase } from '@workbench/canvas-engine/capabilities';
import type { ToolId } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import { ColorSampleController } from './colorSampleController';

/** The controller over a recorded tool; `heldPick` is a press the held tool ends when a request arrives. */
const createHarness = (initialTool: ToolId = 'brush') => {
  const tool = { current: initialTool };
  const routed: [string, ColorSamplePhase][] = [];
  const heldPick = { active: false };
  const controller: ColorSampleController = new ColorSampleController({
    // Ending a held picker cancels its pick, as the pointer pipeline does.
    endHeldTool: () => {
      if (heldPick.active) {
        heldPick.active = false;
        controller.route('#000000', 'cancel');
      }
    },
    getActiveToolId: () => tool.current,
    setTool: (toolId) => {
      tool.current = toolId;
    },
  });
  controller.setRouter((hex, phase) => {
    routed.push([hex, phase]);
    return true;
  });
  const request = () => {
    const result: { hex?: string | null } = {};
    void controller.requestSample().then((hex) => {
      result.hex = hex;
    });
    return result;
  };
  return { controller, heldPick, request, routed, tool };
};

const settled = () => Promise.resolve();

describe('ColorSampleController', () => {
  it('arms the picker and settles a request on the release commit, returning to the previous tool', async () => {
    const h = createHarness('brush');
    const result = h.request();
    expect(h.tool.current).toBe('colorPicker');

    expect(h.controller.route('#111111', 'preview')).toBe(true);
    expect(h.controller.route('#222222', 'preview')).toBe(true);
    await settled();
    expect(result.hex).toBeUndefined();

    expect(h.controller.route('#222222', 'commit')).toBe(true);
    await settled();
    expect(result.hex).toBe('#222222');
    expect(h.tool.current).toBe('brush');
    expect(h.routed).toEqual([]);
  });

  it('keeps a request armed through a cancelled pick and settles it from the next one', async () => {
    const h = createHarness();
    const result = h.request();
    h.controller.route('#111111', 'preview');
    h.controller.route('#111111', 'cancel');
    await settled();
    expect(result.hex).toBeUndefined();

    h.controller.route('#333333', 'preview');
    h.controller.route('#333333', 'commit');
    await settled();
    expect(result.hex).toBe('#333333');
  });

  it('sends a pick’s commit or cancel only to the sink that took its first preview', async () => {
    const h = createHarness('colorPicker');
    // A standing pick is under way when a request arrives: ending the held picker cancels the router's pick, once.
    h.controller.route('#111111', 'preview');
    h.heldPick.active = true;
    const first = h.request();
    expect(h.routed).toEqual([
      ['#111111', 'preview'],
      ['#000000', 'cancel'],
    ]);
    h.controller.route('#444444', 'preview');
    h.controller.route('#444444', 'commit');
    await settled();
    expect(first.hex).toBe('#444444');

    // A request settled by Escape mid-pick keeps that pick's cancel away from the router.
    const second = h.request();
    h.controller.route('#555555', 'preview');
    expect(h.controller.onEscape()).toBe(true);
    expect(h.controller.route('#555555', 'cancel')).toBe(true);
    await settled();
    expect(second.hex).toBeNull();
    expect(h.routed).toHaveLength(2);

    // The other way round: a pick the router took keeps its commit when a request arms mid-pick.
    h.controller.route('#666666', 'preview');
    const third = h.request();
    h.controller.route('#666666', 'commit');
    await settled();
    expect(third.hex).toBeUndefined();
    expect(h.routed.at(-1)).toEqual(['#666666', 'commit']);
  });

  it('declines a pick with no request and no router, so the picker falls back to the brush color', () => {
    const h = createHarness();
    const remove = h.controller.setRouter(() => true);
    remove();
    // The harness router was replaced, so removing the newer one leaves none.
    expect(h.controller.route('#111111', 'preview')).toBe(false);
  });

  it('removes a router only while it is still the installed one', () => {
    const h = createHarness();
    const routedByLater: ColorSamplePhase[] = [];
    const removeEarlier = h.controller.setRouter(() => true);
    h.controller.setRouter((_hex, phase) => {
      routedByLater.push(phase);
      return true;
    });
    removeEarlier();

    expect(h.controller.route('#111111', 'preview')).toBe(true);
    expect(routedByLater).toEqual(['preview']);
  });

  it('settles with null, keeping the requested tool, on a switch off the picker; a newer request supersedes', async () => {
    const h = createHarness('brush');
    const first = h.request();
    const second = h.request();
    await settled();
    expect(first.hex).toBeNull();

    h.controller.onToolChange('colorPicker');
    await settled();
    expect(second.hex).toBeUndefined();
    h.controller.onToolChange('move');
    await settled();
    expect(second.hex).toBeNull();
    // The switch the caller requested wins; the controller restores nothing.
    expect(h.tool.current).toBe('colorPicker');
  });

  it('leaves Escape to the next handler when nothing is armed', () => {
    expect(createHarness().controller.onEscape()).toBe(false);
  });

  it('settles an armed request on dispose, idempotently, and answers later requests with null', async () => {
    const h = createHarness();
    const armed = h.request();
    h.controller.dispose();
    h.controller.dispose();
    await settled();
    expect(armed.hex).toBeNull();

    const late = h.request();
    await settled();
    expect(late.hex).toBeNull();
    expect(h.controller.route('#111111', 'preview')).toBe(false);
  });
});
