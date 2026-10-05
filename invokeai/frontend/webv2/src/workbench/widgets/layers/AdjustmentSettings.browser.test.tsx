import type {
  CanvasAdjustmentEntry,
  CanvasRasterLayerContractV2,
  PreparedDocumentEdit,
} from '@workbench/canvas-engine/api';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createDocumentModel } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { act, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { type AdjustmentsEngine, AdjustmentSettings } from './AdjustmentSettings';
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

const initialEntries = (): CanvasAdjustmentEntry[] => [
  { brightness: 0.1, contrast: 0, id: 'bc1', isEnabled: true, type: 'brightness-contrast' },
  { curves: {}, id: 'cv1', isEnabled: true, type: 'curves' },
  { id: 'hue1', isEnabled: true, rotation: 0, type: 'hue' },
  { gamma: 1, id: 'lv1', inBlack: 0, inWhite: 255, isEnabled: true, outBlack: 0, outWhite: 255, type: 'levels' },
];

const createLayer = (): CanvasRasterLayerContractV2 =>
  ({
    adjustments: initialEntries(),
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

const commits: PreparedDocumentEdit[] = [];
let latestAdjustments: unknown = 'untouched';

const createGroupOwner = () =>
  ({
    adjustments: initialEntries(),
    children: [],
    id: 'group-1',
    isEnabled: true,
    isLocked: false,
    name: 'Group 1',
    type: 'group',
  }) as unknown as CanvasRasterLayerContractV2;

const Harness = ({ entryId, owner = 'raster' }: { entryId: string; owner?: 'raster' | 'group' }) => {
  const [layer, setLayer] = useState(owner === 'group' ? createGroupOwner : createLayer);

  const engine = useMemo(() => {
    const apply = (mutation: CanvasProjectMutation): boolean => {
      const candidate = mutation as { type: string; config?: { adjustments?: unknown } };
      if (candidate.type === 'updateCanvasLayerConfig') {
        latestAdjustments = candidate.config?.adjustments;
        setLayer(
          (current) => ({ ...current, adjustments: candidate.config?.adjustments }) as CanvasRasterLayerContractV2
        );
      }
      return true;
    };

    return {
      document: {
        model: () =>
          createDocumentModel(
            { ...createEmptyCanvasDocument(), stacks: stacksFrom([layer]), selectedLayerId: layer.id },
            { editRevision: 0, projectId: 'test-project' }
          ),
      },
      layers: {
        beginStructuralPreview: () => ({
          apply,
          cancel: (restore?: CanvasProjectMutation) => {
            if (restore) {
              apply(restore);
            }
          },
          commit: (_label: string, edit: PreparedDocumentEdit) => {
            commits.push(edit);
            return { status: apply(edit.forward) ? ('committed' as const) : ('dispatch-rejected' as const) };
          },
        }),
        commitPrepared: (_label: string, edit: PreparedDocumentEdit) => {
          commits.push(edit);
          return { status: apply(edit.forward) ? ('committed' as const) : ('dispatch-rejected' as const) };
        },
      },
    } as unknown as AdjustmentsEngine;
    // Rebuilt per layer change so the stub's model reflects the previewed layer, as the engine's does.
  }, [layer]);

  return <AdjustmentSettings engine={engine} entryId={entryId} layer={layer} />;
};

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 30);
    });
  });

const render = async (entryId = 'cv1', owner: 'raster' | 'group' = 'raster') => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness entryId={entryId} owner={owner} />
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
  commits.length = 0;
  latestAdjustments = 'untouched';
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const sliderTracks = (): HTMLElement[] => Array.from(host!.querySelectorAll<HTMLElement>('[data-part="track"]'));

/** A pointer press-and-release on a slider track at `fraction` of its width. */
const pressTrack = async (track: HTMLElement, fraction: number): Promise<void> => {
  const rect = track.getBoundingClientRect();
  const x = rect.left + rect.width * fraction;
  const y = rect.top + rect.height / 2;
  await settle(() => pointer(track, 'pointerdown', x, y));
  await settle(() => pointer(track, 'pointerup', x, y));
};

const forwardEntries = (edit: PreparedDocumentEdit): CanvasAdjustmentEntry[] =>
  (edit.forward as unknown as { config: { adjustments: CanvasAdjustmentEntry[] } }).config.adjustments;

describe('group-owned adjustment editors', () => {
  it('previews and commits through the group config arm', async () => {
    await render('hue1', 'group');
    const [track] = sliderTracks();
    expect(track).toBeDefined();
    await pressTrack(track!, 0.75);

    expect(commits).toHaveLength(1);
    const forward = commits[0]!.forward as unknown as { config: { layerType: string }; id: string };
    expect(forward.config.layerType).toBe('group');
    expect(forward.id).toBe('group-1');
    expect((forwardEntries(commits[0]!)[2] as { rotation: number }).rotation).toBeCloseTo(90, -1);
  });
});

describe('hue and levels editors', () => {
  it('commits a hue rotation as one whole-stack patch', async () => {
    await render('hue1');
    const [track] = sliderTracks();
    expect(track).toBeDefined();
    await pressTrack(track!, 0.75);

    expect(commits).toHaveLength(1);
    const forward = forwardEntries(commits[0]!);
    expect(forward[2]).toMatchObject({ id: 'hue1', type: 'hue' });
    expect((forward[2] as { rotation: number }).rotation).toBeCloseTo(90, -1);
    // Untouched siblings ride along unchanged.
    expect(forward[0]).toEqual(initialEntries()[0]);
  });

  it('commits a channel scope change as one whole-stack patch keeping the remap values', async () => {
    await render('lv1');
    const trigger = host!.querySelector<HTMLElement>(
      '[aria-label="widgets.layers.adjustments.channel"] [data-part="trigger"], [data-part="trigger"]'
    )!;
    await settle(() => trigger.click());
    const option = [...document.querySelectorAll<HTMLElement>('[data-part="item"]')].find(
      (el) => el.textContent?.trim() === 'widgets.layers.adjustments.channels.r'
    )!;
    expect(option).toBeDefined();
    await settle(() => option.click());

    expect(commits).toHaveLength(1);
    const entry = forwardEntries(commits[0]!)[3] as Extract<CanvasAdjustmentEntry, { type: 'levels' }>;
    expect(entry).toMatchObject({ channel: 'r', gamma: 1, id: 'lv1', inBlack: 0, inWhite: 255 });
  });

  it('commits a levels input-range change while the other fields keep their values', async () => {
    await render('lv1');
    const tracks = sliderTracks();
    expect(tracks).toHaveLength(3);
    await pressTrack(tracks[0]!, 0.25);

    expect(commits).toHaveLength(1);
    const entry = forwardEntries(commits[0]!)[3] as Extract<CanvasAdjustmentEntry, { type: 'levels' }>;
    expect(entry).toMatchObject({ gamma: 1, id: 'lv1', inWhite: 255, outBlack: 0, outWhite: 255 });
    expect(entry.inBlack).toBeGreaterThan(50);
    expect(entry.inBlack).toBeLessThan(80);
  });
});

describe('curves entry editor', () => {
  it('moves a handle while dragging and commits the whole stack with the pre-gesture inverse', async () => {
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
    expect(commits).toHaveLength(1);
    const edit = commits[0]!;
    expect(edit.inverse).toMatchObject({ id: 'layer-1', type: 'updateCanvasLayerConfig' });
    expect((edit.inverse as { config: { adjustments?: unknown } }).config.adjustments).toEqual(initialEntries());
    const forward = (edit.forward as unknown as { config: { adjustments: CanvasAdjustmentEntry[] } }).config
      .adjustments;
    // The untouched sibling entry rides along unchanged; the curves entry carries the new points.
    expect(forward[0]).toEqual(initialEntries()[0]);
    expect(forward[1]).toMatchObject({ id: 'cv1', type: 'curves' });
    expect((forward[1] as { curves: { r?: unknown[] } }).curves.r).toBeDefined();
  });

  it('records nothing and restores the stack when a drag ends where it started', async () => {
    const svg = await render();
    const target = handles(svg).at(-1)!;
    const before = Number(target.getAttribute('cy'));
    const start = centreOf(target);

    await settle(() => pointer(target, 'pointerdown', start.x, start.y));
    await settle(() => pointer(target, 'pointermove', start.x, start.y + 40));
    await settle(() => pointer(target, 'pointermove', start.x, start.y));
    await settle(() => pointer(target, 'pointerup', start.x, start.y));

    expect(Number(handles(svg).at(-1)!.getAttribute('cy'))).toBeCloseTo(before, 5);
    expect(commits).toHaveLength(0);
    expect(latestAdjustments).toEqual(initialEntries());
  });

  it('restores the pre-drag stack and records nothing when the drag is cancelled', async () => {
    const svg = await render();
    const target = handles(svg).at(-1)!;
    const before = Number(target.getAttribute('cy'));
    const start = centreOf(target);

    await settle(() => pointer(target, 'pointerdown', start.x, start.y));
    await settle(() => pointer(target, 'pointermove', start.x, start.y + 40));
    await settle(() => pointer(target, 'pointercancel', start.x, start.y + 40));

    expect(Number(handles(svg).at(-1)!.getAttribute('cy'))).toBeCloseTo(before, 5);
    expect(commits).toHaveLength(0);
    expect(latestAdjustments).toEqual(initialEntries());
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
