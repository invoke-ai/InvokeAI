/**
 * Deleting media a project uses breaks authored work, so the deletion must warn even with confirmations off. The scan
 * fails safe: any project field not given a surface or an exclusion below counts as `otherProjectState`.
 */

import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageSurface } from '@features/gallery/mediaUsage';
import type { Project } from '@workbench/projectContracts';

import { toGalleryItemKey } from '@features/gallery/contracts';
import { collectMediaNames, MAX_HELD_NAME_LENGTH, type MediaNameRefs } from '@workbench/mediaReferences';
import { getProjectWidgetValues } from '@workbench/widgetState';

export interface MediaUsage {
  surface: GalleryMediaUsageSurface;
  label?: string;
  /** Identifies it across copies of the project, as names need not. */
  sourceId?: string;
  /** In selection order. */
  refs: GalleryItemRef[];
}

// Vector guides have no raster sources or media references.
type StackKind = Exclude<keyof Project['canvas']['document']['stacks'], 'vector'>;

/** A new stack kind fails to compile until it has a surface. */
const STACK_SURFACES = {
  raster: 'canvasRaster',
  control: 'canvasControl',
  regional_guidance: 'canvasRegionalGuidance',
  inpaint_mask: 'canvasInpaintMask',
} as const satisfies Record<StackKind, GalleryMediaUsageSurface>;

const STACK_KINDS = Object.keys(STACK_SURFACES) as StackKind[];

/** Typed against `Project`, so a new field fails to compile until it is placed here. */
const PROJECT_FIELDS = {
  // The surfaces read only part of it; the rest, later additions included, is scanned.
  canvas: 'scanned',
  // Session-only; names every generated image.
  events: 'excluded',
  id: 'scanned',
  invocation: 'scanned',
  lastFloatingGeometry: 'scanned',
  // Written back while the project waits for the editor to migrate it.
  droppedArrangements: 'excluded',
  adoptedArrangements: 'excluded',
  heldLegacyLayout: 'scanned',
  legacyLayout: 'scanned',
  name: 'scanned',
  promptHistory: 'scanned',
  // Session-only; names every generated image.
  queue: 'excluded',
  settings: 'scanned',
  // Session-only, never saved.
  undoRedo: 'excluded',
  // A record of a past run: the next run compiles from the widget's values.
  widgetGraphs: 'excluded',
  // Scanned without Gallery instances (below).
  widgetInstances: 'scanned',
  views: 'scanned',
  // Session-only, never saved.
  workflowHistories: 'excluded',
  // Scanned without each workflow's last run (below).
  workflows: 'scanned',
  workspaceId: 'scanned',
} as const satisfies Record<keyof Project, 'excluded' | 'scanned'>;

const isExcludedField = (key: string): boolean =>
  Object.hasOwn(PROJECT_FIELDS, key) && PROJECT_FIELDS[key as keyof Project] === 'excluded';

const scan = (roots: readonly unknown[]): MediaNameRefs => collectMediaNames(roots, () => false, MAX_HELD_NAME_LENGTH);

const selectedIn = (refs: MediaNameRefs, selected: readonly GalleryItemRef[]): GalleryItemRef[] =>
  selected.filter((ref) => (ref.kind === 'image' ? refs.images : refs.videos).has(ref.name));

const without = <Value extends object>(record: Value, keys: readonly PropertyKey[]): Partial<Value> =>
  Object.fromEntries(Object.entries(record).filter(([key]) => !keys.includes(key))) as Partial<Value>;

/** Gallery instances only show media and workflows' last runs only record output, so both are excluded. */
const remainderRoots = (project: Project): unknown[] => {
  const { canvas, widgetInstances, workflows } = project;
  const rest: Record<string, unknown> = {
    ...project,
    canvas: {
      ...canvas,
      document: { ...canvas.document, stacks: without(canvas.document.stacks, STACK_KINDS) },
      snapshots: canvas.snapshots.map((snapshot) => without(snapshot, ['document', 'stagedImages'])),
      stagingArea: without(canvas.stagingArea, ['pendingImages']),
    },
    widgetInstances: Object.values(widgetInstances).filter((instance) => instance.typeId !== 'gallery'),
    workflows: {
      ...workflows,
      entries: workflows.entries.map(({ document, lastRun: _lastRun, ...entry }) => ({
        ...entry,
        document: without(document, ['nodes']),
      })),
    },
  };

  return Object.entries(rest).flatMap(([key, value]) => (isExcludedField(key) ? [] : [value]));
};

/** Staged canvas results and snapshots count: the user can still accept or restore them. */
export const collectMediaUsage = (project: Project, selected: readonly GalleryItemRef[]): MediaUsage[] => {
  if (selected.length === 0) {
    return [];
  }

  const usage: MediaUsage[] = [];
  const reported = new Set<string>();
  const add = (
    surface: GalleryMediaUsageSurface,
    roots: readonly unknown[],
    source?: { id: string; label: string }
  ): void => {
    const refs = selectedIn(scan(roots), selected);

    if (refs.length > 0) {
      refs.forEach((ref) => reported.add(toGalleryItemKey(ref)));
      usage.push(source ? { label: source.label, refs, sourceId: source.id, surface } : { refs, surface });
    }
  };

  add('generateReferenceImages', [getProjectWidgetValues(project, 'generate').referenceImages]);

  for (const kind of STACK_KINDS) {
    add(STACK_SURFACES[kind], [project.canvas.document.stacks[kind]]);
  }
  add('canvasStaging', [project.canvas.stagingArea.pendingImages]);

  for (const snapshot of project.canvas.snapshots) {
    add('canvasSnapshot', [snapshot.document, snapshot.stagedImages], { id: snapshot.id, label: snapshot.name });
  }

  add('upscale', [getProjectWidgetValues(project, 'upscale').inputImage]);
  add('video', [getProjectWidgetValues(project, 'video')]);

  for (const entry of project.workflows.entries) {
    add('workflow', [entry.document.nodes], { id: entry.document.id, label: entry.document.name });
  }

  const elsewhere = selectedIn(scan(remainderRoots(project)), selected).filter(
    (ref) => !reported.has(toGalleryItemKey(ref))
  );

  if (elsewhere.length > 0) {
    usage.push({ refs: elsewhere, surface: 'otherProjectState' });
  }

  return usage;
};

const SURFACE_ORDER = {
  generateReferenceImages: 0,
  canvasRaster: 1,
  canvasControl: 2,
  canvasRegionalGuidance: 3,
  canvasInpaintMask: 4,
  canvasStaging: 5,
  canvasSnapshot: 6,
  upscale: 7,
  video: 8,
  workflow: 9,
  otherProjectState: 10,
} as const satisfies Record<GalleryMediaUsageSurface, number>;

/** Surfaces match by id across copies; `otherProjectState` keeps only refs no listed surface of either reports. */
export const mergeMediaUsage = (first: readonly MediaUsage[], second: readonly MediaUsage[]): MediaUsage[] => {
  const merged = new Map<string, MediaUsage>();

  for (const entry of [...first, ...second]) {
    const key = JSON.stringify([entry.surface, entry.sourceId ?? entry.label ?? null]);
    const existing = merged.get(key);

    if (!existing) {
      merged.set(key, { ...entry, refs: [...entry.refs] });
      continue;
    }

    const held = new Set(existing.refs.map(toGalleryItemKey));

    existing.refs.push(...entry.refs.filter((ref) => !held.has(toGalleryItemKey(ref))));
  }

  const listed = new Set(
    [...merged.values()]
      .filter((entry) => entry.surface !== 'otherProjectState')
      .flatMap((entry) => entry.refs.map(toGalleryItemKey))
  );

  return [...merged.values()]
    .map((entry) =>
      entry.surface === 'otherProjectState'
        ? { ...entry, refs: entry.refs.filter((ref) => !listed.has(toGalleryItemKey(ref))) }
        : entry
    )
    .filter((entry) => entry.refs.length > 0)
    .sort((a, b) => SURFACE_ORDER[a.surface] - SURFACE_ORDER[b.surface]);
};
