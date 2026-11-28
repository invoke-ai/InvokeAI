/** Test-only: the registry's availability and seeded generators of structures and views. */
import type { FloatingWidgetMode, WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetTypeId } from '@workbench/widgetContracts';

import { WIDGET_REGIONS } from '@workbench/layoutContracts';
import { layoutPresets } from '@workbench/layoutPresets';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';

import type { LayoutAvailability } from './resolveLayout';
import type { StructureAction } from './structureReducer';
import type { WorkspaceView } from './view';

import { createRegistryLayoutAvailability } from './registryAvailability';
import { normalizeStructure, type WorkspaceStructure } from './structure';
import { reduceStructure } from './structureReducer';
import {
  frontRegion,
  getGuestInstanceId,
  raiseWindow,
  setRegionCollapsed,
  setRegionSize,
  setWindowGeometry,
  setWindowMode,
  showGuest,
} from './view';

export const builtInStructures: readonly WorkspaceStructure[] = layoutPresets.map(({ snapshot }) =>
  normalizeStructure(snapshot)
);

export const registryAvailability: LayoutAvailability = createRegistryLayoutAvailability();

export const withUnavailable = (
  availability: LayoutAvailability,
  typeIds: readonly WidgetTypeId[]
): LayoutAvailability => ({
  ...availability,
  getPresentation: (typeId, region) => (typeIds.includes(typeId) ? null : availability.getPresentation(typeId, region)),
});

/** Mulberry32: a small seeded generator, so a failing case replays from its seed. */
export const createRandom = (seed: number) => {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);

    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  const int = (max: number) => Math.floor(next() * max);
  const pick = <Value>(values: readonly Value[]): Value => values[int(values.length)]!;

  return { int, next, pick };
};

type Random = ReturnType<typeof createRandom>;

const MODES: readonly FloatingWidgetMode[] = ['windowed', 'maximized', 'shaded'];

const typesFor = (region: WidgetRegion) => getWidgetsForRegion(region).map(({ manifest }) => manifest.id);

const randomGeometry = (random: Random) => ({
  heightPx: random.int(900),
  widthPx: random.int(1200),
  x: random.int(1600) - 200,
  y: random.int(1000) - 100,
});

/** A structure action over what `structure` holds, with ids sometimes stale. */
export const randomStructureAction = (random: Random, structure: WorkspaceStructure, step: number): StructureAction => {
  const region = random.pick(WIDGET_REGIONS);
  const instanceIds = [...Object.keys(structure.widgetInstances), 'ghost'];
  const instanceId = random.pick(instanceIds);
  const members = structure.widgetRegions[region].instanceIds;

  switch (random.int(9)) {
    case 0:
    case 1:
      return {
        createNew: random.int(4) === 0,
        index: random.int(6),
        mintId: `minted-${step}`,
        region,
        type: 'addWidget',
        typeId: random.pick(typesFor(region)),
      };
    case 2:
      return { instanceId: random.pick([...members, instanceId]), region, type: 'removeWidget' };
    case 3:
      return {
        fromRegion: region,
        instanceId: random.pick([...members, instanceId]),
        toIndex: random.int(6),
        toRegion: random.pick(WIDGET_REGIONS),
        type: 'moveWidget',
      };
    case 4:
      return { instanceIds: [...members].sort(() => random.next() - 0.5), region, type: 'reorderWidgets' };
    case 5:
      return { align: random.pick(['start', 'end'] as const), instanceId, region: 'bottom', type: 'setAlignment' };
    case 6:
      return { geometry: random.int(2) ? randomGeometry(random) : undefined, instanceId, type: 'floatWidget' };
    case 7:
      return {
        instanceId: random.pick([...Object.keys(structure.floatingWidgets ?? {}), instanceId]),
        type: 'dockWindow',
      };
    default:
      return {
        instanceId: random.pick([...Object.keys(structure.floatingWidgets ?? {}), instanceId]),
        type: 'closeWindow',
      };
  }
};

export const generateStructure = (
  random: Random,
  availability: LayoutAvailability,
  steps: number
): WorkspaceStructure => {
  let structure = random.pick(builtInStructures);

  for (let step = 0; step < steps; step += 1) {
    structure = reduceStructure(structure, randomStructureAction(random, structure, step), availability);
  }

  if (random.int(3) > 0) {
    return structure;
  }

  // A stored or imported structure may place a type where it cannot show, which no edit would; repair 1 hides it.
  const region = random.pick(WIDGET_REGIONS);
  const typeId = random.pick(
    [...new Set(WIDGET_REGIONS.flatMap(typesFor))].filter((type) => !typesFor(region).includes(type))
  );
  const instanceId = `unallowed-${typeId}`;

  return normalizeStructure({
    ...structure,
    widgetInstances: { ...structure.widgetInstances, [instanceId]: { id: instanceId, typeId } },
    widgetRegions: {
      ...structure.widgetRegions,
      [region]: {
        ...structure.widgetRegions[region],
        instanceIds: [...structure.widgetRegions[region].instanceIds, instanceId],
      },
    },
  });
};

/** A view built by the writers, then given stale or ill-fitting entries no writer would store. */
export const generateView = (random: Random, structure: WorkspaceStructure): WorkspaceView | undefined => {
  const typeIds = [...new Set(WIDGET_REGIONS.flatMap(typesFor))];
  let view: WorkspaceView | undefined;

  for (let step = random.int(8); step > 0; step -= 1) {
    const region = random.pick(WIDGET_REGIONS);
    const windowIds = [...Object.keys(structure.floatingWidgets ?? {}), ...Object.keys(view?.guests ?? {})];
    const windowId = windowIds.length > 0 ? random.pick(windowIds) : 'ghost';

    switch (random.int(7)) {
      case 6: {
        // Types and ids the structure already has too: the resolver must not show what it supersedes.
        const structureIds = Object.keys(structure.widgetInstances);
        // Often the type of a placement repair 1 hides, when there is one: it must not keep the guest away.
        const hiddenIds = structureIds.filter((id) => id.startsWith('unallowed-'));
        const typeId =
          hiddenIds.length > 0 && random.int(2) === 0
            ? structure.widgetInstances[random.pick(hiddenIds)]!.typeId
            : random.int(2)
              ? random.pick(typeIds)
              : (structure.widgetInstances[random.pick(structureIds)]?.typeId ?? random.pick(typeIds));
        const instanceId = random.int(4) === 0 ? random.pick(structureIds) : getGuestInstanceId(structure, typeId, []);

        view = showGuest(
          view,
          structure,
          instanceId,
          { surface: random.pick(['center', 'window'] as const), typeId },
          randomGeometry(random)
        );
        break;
      }
      case 0:
        view = frontRegion(
          view,
          structure,
          region,
          random.pick([...structure.widgetRegions[region].instanceIds, 'ghost']),
          registryAvailability
        );
        break;
      case 1:
        view = setRegionSize(view, structure, region, random.int(1000));
        break;
      case 2:
        view = setRegionCollapsed(view, structure, region, random.int(2) === 0);
        break;
      case 3:
        view = setWindowGeometry(view, structure, windowId, randomGeometry(random));
        break;
      case 4:
        view = setWindowMode(view, structure, windowId, random.pick(MODES));
        break;
      default:
        view = raiseWindow(view, structure, windowId);
    }
  }

  if (random.int(3) > 0) {
    return view;
  }

  const anyId = () => random.pick([...Object.keys(structure.widgetInstances), 'ghost', 'stale']);

  return {
    ...view,
    fronts: { ...view?.fronts, [random.pick(WIDGET_REGIONS)]: anyId() },
    guests: {
      ...view?.guests,
      [anyId()]: { surface: random.pick(['center', 'window'] as const), typeId: random.pick(typeIds) },
    },
    sizes: { ...view?.sizes, [random.pick(['left', 'right', 'bottom'] as const)]: random.int(2000) - 500 },
    windows: { ...view?.windows, [anyId()]: { heightPx: random.int(50), widthPx: random.int(50) } },
  };
};

/** Some registered types made unavailable, as a disabled or admin-only widget is; sometimes every center view. */
export const generateAvailability = (random: Random, structure: WorkspaceStructure): LayoutAvailability => {
  const typeIds = [...new Set(WIDGET_REGIONS.flatMap(typesFor))];
  const centerTypeIds = structure.widgetRegions.center.instanceIds.map(
    (instanceId) => structure.widgetInstances[instanceId]?.typeId ?? ''
  );

  return withUnavailable(registryAvailability, [
    ...typeIds.filter(() => random.int(6) === 0),
    ...(random.int(4) === 0 ? centerTypeIds : []),
  ]);
};
