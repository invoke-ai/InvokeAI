import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';

import { ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AdjustmentsPopover } from './AdjustmentsPopover';
import { CURVE_SIZE } from './curveEditorMath';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

// The popover calls this hook for its no-engine fallback path. The harness
// drives edits through a stand-in engine instead, so the dispatch is never used
// and can stay a stable no-op rather than something reassigned during render.
const noopDispatch = (): void => undefined;
vi.mock('@workbench/useCanvasProjectMutationDispatch', () => ({
  useCanvasProjectMutationDispatch: () => noopDispatch,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const createLayer = (): CanvasRasterLayerContractV2 =>
  ({
    blendMode: 'normal',
    id: 'layer-1',
    isEnabled: true,
    isLocked: false,
    name: 'Layer 1',
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  }) as unknown as CanvasRasterLayerContractV2;

const Harness = () => {
  const [layer, setLayer] = useState(createLayer);

  // Stands in for the canvas engine: both the live preview and the committed
  // edit land back on the layer prop, which is what the editor re-renders from.
  const engine = useMemo(() => {
    const apply = (mutation: CanvasProjectMutation): boolean => {
      const candidate = mutation as { type: string; config?: { adjustments?: unknown } };
      if (candidate.type === 'updateCanvasLayerConfig') {
        setLayer(
          (current) => ({ ...current, adjustments: candidate.config?.adjustments }) as CanvasRasterLayerContractV2
        );
      }
      return true;
    };

    return {
      layers: {
        applyStructuralPreview: apply,
        commitStructural: (_label: string, forward: CanvasProjectMutation) => apply(forward),
      },
    } as unknown as CanvasStructuralEngine;
  }, []);

  return <AdjustmentsPopover engine={engine} layer={layer} />;
};

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 30);
    });
  });

const render = async () => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  // Narrower than the editor's intrinsic size, so the square-fit path is real.
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  return host.querySelector<SVGSVGElement>(`svg[viewBox="0 0 ${CURVE_SIZE} ${CURVE_SIZE}"]`)!;
};

const handles = (svg: SVGSVGElement): SVGCircleElement[] => Array.from(svg.querySelectorAll('circle'));

const centreOf = (element: Element): { x: number; y: number } => {
  const rect = element.getBoundingClientRect();
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
};

const pointer = (target: Element, type: string, x: number, y: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX: x, clientY: y, isPrimary: true, pointerId: 1 })
  );
};

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('curves editor', () => {
  it('paints its chrome from resolved theme tokens', async () => {
    // Hand-written `var(--chakra-colors-bg-inset)` did not exist (Chakra emits
    // `--chakra-colors-bg.inset`), so `fill` fell back to black and `stroke` to
    // `none` — the whole editor rendered as a black box.
    const svg = await render();
    const area = svg.querySelector('rect')!;
    const curve = svg.querySelector('path')!;

    for (const [label, value] of [
      ['area fill', getComputedStyle(area).fill],
      ['curve stroke', getComputedStyle(curve).stroke],
      ['handle fill', getComputedStyle(handles(svg)[0]!).fill],
    ] as const) {
      expect(value, label).not.toBe('none');
      expect(value, label).not.toBe('rgb(0, 0, 0)');
    }

    expect(getComputedStyle(svg).backgroundColor).not.toBe('rgba(0, 0, 0, 0)');
  });

  it('makes nothing in the editor selectable, so a drag cannot become a text drag', async () => {
    // Double-clicking to add a point is a word-select gesture; without this it
    // latched onto the channel select's label and the next handle press dragged
    // that selection instead, trailing a floating "Red" across the screen.
    const svg = await render();
    const editor = svg.parentElement!;

    expect(getComputedStyle(editor).userSelect).toBe('none');
    expect(getComputedStyle(svg).userSelect).toBe('none');
    expect(getComputedStyle(editor.querySelector('button')!).userSelect).toBe('none');
  });

  it('keeps the drawing square so pointer coordinates map onto the viewBox', async () => {
    const svg = await render();
    const rect = svg.getBoundingClientRect();

    expect(rect.width).toBeCloseTo(rect.height, 0);
  });

  it('moves a handle while dragging it', async () => {
    const svg = await render();
    const target = handles(svg).at(-1)!;
    const before = Number(target.getAttribute('cy'));
    const start = centreOf(target);

    await settle(() => pointer(target, 'pointerdown', start.x, start.y));
    await settle(() => pointer(target, 'pointermove', start.x, start.y + 40));

    const during = Number(handles(svg).at(-1)!.getAttribute('cy'));
    expect(during).toBeGreaterThan(before);

    await settle(() => pointer(target, 'pointerup', start.x, start.y + 40));
    expect(Number(handles(svg).at(-1)!.getAttribute('cy'))).toBeCloseTo(during, 5);
  });

  it('adds a point under the pointer rather than offset from it', async () => {
    // Guards the letterboxing bug: a non-square box mapped pointer x across the
    // whole element while the viewBox only occupied part of it.
    const svg = await render();
    const rect = svg.getBoundingClientRect();
    // Deliberately off centre: a letterboxed viewBox still maps the midpoint
    // correctly, because the dead space either side cancels out there.
    const targetX = rect.left + rect.width * 0.25;

    await settle(() =>
      svg.dispatchEvent(
        new MouseEvent('dblclick', { bubbles: true, clientX: targetX, clientY: rect.top + rect.height * 0.5 })
      )
    );

    expect(handles(svg)).toHaveLength(3);
    const added = centreOf(handles(svg)[1]!);
    expect(added.x).toBeCloseTo(targetX, 0);
  });
});
