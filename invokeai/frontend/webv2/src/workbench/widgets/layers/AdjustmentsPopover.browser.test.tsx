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
    const svg = await render();
    const rect = svg.getBoundingClientRect();
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
