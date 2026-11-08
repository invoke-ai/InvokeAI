/**
 * Where a project still uses gallery media the user is about to delete. Deleting such media removes the backing
 * file under authored work, so the deletion flow must warn even when ordinary confirmations are turned off.
 *
 * The scan fails safe. The surfaces the panel names are listed below; whatever else in the project names a selected
 * item counts too, as `otherProjectState`, unless an exclusion below says it only shows or records media. A field this
 * build hydrates into `Project` that stores a media name therefore warns until it is given a surface or an exclusion.
 * The scan sees only what loading kept, though: a field only a newer build writes is dropped when this build loads the
 * project (the document allowlist, and the workspace canvas, of which only the selected layer and staging area are
 * kept), so it is not scanned here.
 */

import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryMediaUsageSurface } from '@features/gallery/mediaUsage';
import type { Project } from '@workbench/projectContracts';

import { toGalleryItemKey } from '@features/gallery/contracts';
import { collectMediaNames, MAX_HELD_NAME_LENGTH, type MediaNameRefs } from '@workbench/mediaReferences';
import { getProjectWidgetValues } from '@workbench/widgetState';

export interface MediaUsage {
  surface: GalleryMediaUsageSurface;
  /** The workflow's or snapshot's name, for the `workflow` and `canvasSnapshot` surfaces. */
  label?: string;
  /** The workflow document's or snapshot's id: what identifies it across copies of the project, as names need not. */
  sourceId?: string;
  /** Selected items this surface uses, in selection order as `collectMediaUsage` lists them. */
  refs: GalleryItemRef[];
}

type StackKind = keyof Project['canvas']['document']['stacks'];

/** The canvas layer stacks, each its own surface; a new stack kind fails to compile until it has one. */
const STACK_SURFACES = {
  raster: 'canvasRaster',
  control: 'canvasControl',
  regional_guidance: 'canvasRegionalGuidance',
  inpaint_mask: 'canvasInpaintMask',
} as const satisfies Record<StackKind, GalleryMediaUsageSurface>;

const STACK_KINDS = Object.keys(STACK_SURFACES) as StackKind[];

/**
 * How the remainder treats each project field: `scanned`, or `excluded` because it only shows or records media, or is
 * session state that is never saved. Typed against `Project`, so a new field fails to compile until it is placed here.
 */
const PROJECT_FIELDS = {
  // The listed surfaces read its layer stacks, staged results and each snapshot's document and staged results; the
  // rest of it, a stack kind or snapshot field added later included, is scanned here.
  canvas: 'scanned',
  // Session-only run log; it names every generated image.
  events: 'excluded',
  floatingWidgets: 'scanned',
  id: 'scanned',
  invocation: 'scanned',
  lastFloatingGeometry: 'scanned',
  layout: 'scanned',
  name: 'scanned',
  presetWorkingLayouts: 'scanned',
  promptHistory: 'scanned',
  // Session-only queue runs and their results; they name every generated image.
  queue: 'excluded',
  settings: 'scanned',
  // Session-only undo history, never saved; restoring it is not work the deletion breaks.
  undoRedo: 'excluded',
  // The last submitted graph per source widget, a record of a past run: the next run compiles from the widget's values.
  widgetGraphs: 'excluded',
  // Scanned without Gallery instances (below).
  widgetInstances: 'scanned',
  widgetRegions: 'scanned',
  // Session-only workflow edit histories, never saved.
  workflowHistories: 'excluded',
  // Scanned without each workflow's last run (below).
  workflows: 'scanned',
} as const satisfies Record<keyof Project, 'excluded' | 'scanned'>;

const isExcludedField = (key: string): boolean =>
  Object.hasOwn(PROJECT_FIELDS, key) && PROJECT_FIELDS[key as keyof Project] === 'excluded';

const scan = (roots: readonly unknown[]): MediaNameRefs => collectMediaNames(roots, () => false, MAX_HELD_NAME_LENGTH);

const selectedIn = (refs: MediaNameRefs, selected: readonly GalleryItemRef[]): GalleryItemRef[] =>
  selected.filter((ref) => (ref.kind === 'image' ? refs.images : refs.videos).has(ref.name));

/** `record` without `keys`: what a listed surface leaves of a container it reads only part of. */
const without = <Value extends object>(record: Value, keys: readonly PropertyKey[]): Partial<Value> =>
  Object.fromEntries(Object.entries(record).filter(([key]) => !keys.includes(key))) as Partial<Value>;

/**
 * The project without its excluded fields and without exactly what the listed surfaces scan (most of its bulk), so
 * anything they do not read, a stack kind or snapshot field added later included, is still scanned. Excluded within
 * scanned fields: Gallery instances (selection, recents, compare image, searches: they show media) and workflows'
 * last runs (a record of a run's output).
 */
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

/**
 * The project's surfaces that use any of `selected`, in panel order: live work, staged canvas results and canvas
 * snapshots, which the user can still accept or restore; then `otherProjectState` for selected items that only the rest
 * of the project names (see `PROJECT_FIELDS` for what that leaves out).
 */
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

/** Panel order, as `collectMediaUsage` lists the surfaces. */
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

/**
 * Two scans of one project (two copies of it) as one, in panel order: a surface either uses keeps the refs of both (a
 * workflow or snapshot matched by id, named as in `first`), and `otherProjectState` keeps only refs no listed surface
 * of either reports.
 */
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
