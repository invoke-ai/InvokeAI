/**
 * Where a project still uses gallery media the user is about to delete. Deleting such media removes the backing
 * file under authored work, so the deletion flow must warn even when ordinary confirmations are turned off.
 */

import type { GalleryItemRef } from '@features/gallery/contracts';
import type { Project } from '@workbench/projectContracts';

import { collectMediaNames, MAX_HELD_NAME_LENGTH, type MediaNameRefs } from '@workbench/mediaReferences';
import { getProjectWidgetValues } from '@workbench/widgetState';

export type MediaUsageSurface =
  | 'generateReferenceImages'
  | 'canvasRaster'
  | 'canvasControl'
  | 'canvasRegionalGuidance'
  | 'canvasInpaintMask'
  | 'upscale'
  | 'video'
  | 'workflow';

export interface MediaUsage {
  surface: MediaUsageSurface;
  /** The workflow's name, for the `workflow` surface. */
  label?: string;
  /** Selected names this surface uses, in selection order. */
  names: string[];
}

type UsageProject = Pick<Project, 'canvas' | 'widgetInstances' | 'workflows'>;

const scan = (roots: readonly unknown[]): MediaNameRefs => collectMediaNames(roots, () => false, MAX_HELD_NAME_LENGTH);

const selectedNamesIn = (refs: MediaNameRefs, selected: readonly GalleryItemRef[]): string[] =>
  selected.filter((ref) => (ref.kind === 'image' ? refs.images : refs.videos).has(ref.name)).map((ref) => ref.name);

/**
 * The project's live surfaces that use any of `selected`, in panel order. History (undo, snapshots, queue) does
 * not count: deleting what only history could restore is a loss, not a hole in current work.
 */
export const collectMediaUsage = (project: UsageProject, selected: readonly GalleryItemRef[]): MediaUsage[] => {
  if (selected.length === 0) {
    return [];
  }

  const usage: MediaUsage[] = [];
  const add = (surface: MediaUsageSurface, roots: readonly unknown[], label?: string): void => {
    const names = selectedNamesIn(scan(roots), selected);

    if (names.length > 0) {
      usage.push(label === undefined ? { names, surface } : { label, names, surface });
    }
  };

  add('generateReferenceImages', [getProjectWidgetValues(project, 'generate').referenceImages]);

  const { stacks } = project.canvas.document;
  add('canvasRaster', [stacks.raster]);
  add('canvasControl', [stacks.control]);
  add('canvasRegionalGuidance', [stacks.regional_guidance]);
  add('canvasInpaintMask', [stacks.inpaint_mask]);

  add('upscale', [getProjectWidgetValues(project, 'upscale').inputImage]);
  add('video', [getProjectWidgetValues(project, 'video')]);

  for (const entry of project.workflows.entries) {
    add('workflow', [entry.document.nodes], entry.document.name);
  }

  return usage;
};
