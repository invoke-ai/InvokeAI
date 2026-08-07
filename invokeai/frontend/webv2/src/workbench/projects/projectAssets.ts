/**
 * Which assets a project document points at, and how to rewrite those pointers.
 *
 * The document stores pixels the way `canvas-engine/contracts.ts` describes:
 * by name, never by URL or inline data. That is what makes the document small
 * enough to autosave, and also what makes a bare document export useless on
 * another install — the names resolve to nothing there. Bundling the bytes into
 * an `.invk` archive means first knowing which names the document actually uses.
 *
 * ### Two kinds, kept apart
 *
 * Images and videos live in separate backend namespaces, with separate tables,
 * separate fetch and upload routes, and no shared name space. A document can
 * carry both — `video_name` reaches `projectGraph` through an imported workflow
 * whose node value was authored elsewhere (`VideoField` in the backend's
 * `fields.py`; webv2 cannot author one yet, but it round-trips one verbatim).
 * So collection returns a set per kind and remapping takes a mapping per kind:
 * an image and a video could in principle share a name, and a single flat map
 * would let one rewrite the other.
 *
 * ### Collection is by key, not by path
 *
 * Three keys in the document hold an asset name: `imageName` (the webv2 canvas
 * contracts), `image_name` and `video_name` (graph node values, which mirror the
 * backend's field naming). Collecting every string found at those keys is
 * complete by construction — a new control-adapter kind, a new node field, a
 * widget nobody has written yet, all get picked up without touching this file.
 *
 * The alternative, walking a hand-written list of paths, is what the legacy
 * frontend does, and it is why its collector enumerates `ip_adapter` and
 * `flux_redux` by name and silently drops the pixels of any adapter added since.
 * A false positive here costs nothing: a string that is not really an asset name
 * fails to resolve on the server and is skipped.
 *
 * ### Collection skips history, remapping does not
 *
 * A project document carries far more than the live document. Every queue entry
 * embeds a whole canvas snapshot, `canvas.snapshots` holds full document copies,
 * and the gallery widget keeps up to sixty recent results. Bundling all of that
 * turns a working project into a multi-gigabyte archive, so
 * {@link collectLiveAssetRefs} walks only what the project needs in order to
 * open correctly.
 *
 * {@link remapAssetRefs} walks everything, with no skips. The asymmetry is the
 * point: if a bundled asset comes back from the server under a new name, every
 * reference to it has to follow — including the ones in history we chose not to
 * bundle, which would otherwise point at the pre-import name forever.
 */

/** Keys whose string values name an image. */
const IMAGE_NAME_KEYS = new Set(['imageName', 'image_name']);

/** Keys whose string values name a video. */
const VIDEO_NAME_KEYS = new Set(['video_name']);

/** Top-level document keys that are history rather than live content. */
export const PROJECT_HISTORY_ROOT_KEYS: ReadonlySet<string> = new Set(['events', 'graphHistory', 'queue']);

/** Keys that introduce history at any depth (`canvas.snapshots`, the gallery's recents). */
export const PROJECT_HISTORY_KEYS: ReadonlySet<string> = new Set(['recentImages', 'snapshot', 'snapshots']);

/**
 * The gallery widget's selection: pointers into gallery content rather than
 * anything the document renders. `selectedImage` holds a whole polymorphic
 * `GalleryItem` (image or video), the two name keys hold `"<kind>:<name>"`, and
 * `compareImage` holds the image the Preview widget is comparing against.
 *
 * All are skipped deliberately. The gallery is per-install content: a project
 * that arrives on another machine should open with nothing selected, not drag a
 * stranger's gallery in behind it.
 *
 * Skipping them during collection is only half of that. A reference that is not
 * bundled and not stripped still travels — it simply travels broken, pointing at
 * an image the receiving server has never had, and import cannot even report it
 * as dangling because the same skip hides it from the restore pass.
 * {@link stripGallerySelection} is the other half, and the export planner
 * applies it before the document is serialized.
 *
 * The two halves got here differently, which is why the skip is by key rather
 * than left to chance. `selectedImage` and the name keys were already missed,
 * but only because `GalleryItem` spells its name field `name` and no collector
 * key matches it — an accident that renaming the field would have undone.
 * `compareImage` is a `GeneratedImageContract`, so it carries `imageName` and
 * was genuinely being bundled: excluding it is a real change, and the right one,
 * since a comparison someone happened to leave open is not part of the project.
 */
export const GALLERY_SELECTION_KEYS: ReadonlySet<string> = new Set([
  'compareImage',
  'selectedImage',
  'selectedImageName',
  'selectedImageNames',
]);

/**
 * Board ids the gallery widget holds: which board this project owns, and which
 * one new results are routed to. Both are *installation* state, not project
 * state — a board id means nothing on the machine the project arrives at.
 *
 * `projectBoardId` in particular is a cache, not a fact. The server owns the
 * project-to-board relationship, and hydration overwrites this from the project
 * record. Exporting it would carry a stale pointer to a board the receiving
 * install has never had, and importing it would fight the authoritative value
 * the create response is about to supply.
 *
 * Board ids elsewhere in the document — a workflow node's board input, say — are
 * semantically meaningful and are deliberately NOT stripped. That is why this
 * set names only the two keys the gallery widget owns.
 */
export const GALLERY_INSTALLATION_KEYS: ReadonlySet<string> = new Set(['projectBoardId', 'selectedBoardId']);

const INSTALLATION_STATE_KEYS: ReadonlySet<string> = new Set([...GALLERY_SELECTION_KEYS, ...GALLERY_INSTALLATION_KEYS]);

/**
 * URLs the document caches beside a media name, which name *this* install's server and this
 * install's copy of the media.
 *
 * They are installation state like the keys above, but they cannot be removed the way those are.
 * The persisted-recents validator (`getBoundedRecentImages`) requires both to be strings and drops
 * any entry missing one, so deleting them would silently discard the whole gallery-recents overlay
 * on import. Blanked instead: every consumer already falls back to deriving the URL from the media
 * name (`item.thumbnailUrl || item.fullUrl`, `slot.candidate.thumbnailUrl || galleryImageUrls…`),
 * which is the only correct answer once a transfer has renamed the media anyway. The name is
 * remapped; a URL built around the *old* name is not, and would keep resolving — to the source
 * project's picture.
 */
const DERIVED_URL_KEYS: ReadonlySet<string> = new Set(['imageUrl', 'thumbnailUrl', 'videoUrl']);

export interface ProjectAssetRefs {
  images: Set<string>;
  videos: Set<string>;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const collectFrom = (node: unknown, refs: ProjectAssetRefs): void => {
  if (Array.isArray(node)) {
    for (const item of node) {
      collectFrom(item, refs);
    }

    return;
  }

  if (!isRecord(node)) {
    return;
  }

  for (const [key, value] of Object.entries(node)) {
    if (PROJECT_HISTORY_KEYS.has(key) || GALLERY_SELECTION_KEYS.has(key)) {
      continue;
    }

    if (typeof value === 'string' && value !== '') {
      if (IMAGE_NAME_KEYS.has(key)) {
        refs.images.add(value);
        continue;
      }

      if (VIDEO_NAME_KEYS.has(key)) {
        refs.videos.add(value);
        continue;
      }
    }

    collectFrom(value, refs);
  }
};

/**
 * Every asset name the project needs in order to open: canvas layers and masks,
 * reference images, widget values, and graph node inputs. History and gallery
 * selections are excluded — see the module docblock.
 */
export const collectLiveAssetRefs = (projectDocument: Record<string, unknown>): ProjectAssetRefs => {
  const refs: ProjectAssetRefs = { images: new Set<string>(), videos: new Set<string>() };

  for (const [key, value] of Object.entries(projectDocument)) {
    if (PROJECT_HISTORY_ROOT_KEYS.has(key)) {
      continue;
    }

    collectFrom(value, refs);
  }

  return refs;
};

const stripNode = (node: unknown): unknown => {
  if (Array.isArray(node)) {
    let hasChanged = false;
    const next = node.map((item) => {
      const stripped = stripNode(item);

      hasChanged ||= stripped !== item;

      return stripped;
    });

    return hasChanged ? next : node;
  }

  if (!isRecord(node)) {
    return node;
  }

  let hasChanged = false;
  const next: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(node)) {
    if (INSTALLATION_STATE_KEYS.has(key)) {
      hasChanged = true;
      continue;
    }

    if (DERIVED_URL_KEYS.has(key) && typeof value === 'string') {
      next[key] = '';
      hasChanged ||= value !== '';
      continue;
    }

    const stripped = stripNode(value);

    next[key] = stripped;
    hasChanged ||= stripped !== value;
  }

  return hasChanged ? next : node;
};

/**
 * Drop everything that describes *this install* rather than the project — the
 * gallery's selection, its board ids, and the URLs cached beside media names — at
 * every depth, so an exported project opens with nothing selected, on the board
 * the receiving server assigns it, resolving its pictures through the names it
 * actually owns.
 *
 * Every reader of the selection and board values already tolerates their absence —
 * selection is parsed with `typeof`/`Array.isArray` guards and falls back to
 * nothing, and the board ids are re-supplied from the project record — so removing
 * the key is the same as clearing it, without inventing a shape. The cached URLs
 * are blanked rather than removed, for the reason {@link DERIVED_URL_KEYS} gives.
 * Subtrees with nothing to change keep their identity.
 *
 * Stripping by key at any depth rather than by walking into `widgetInstances` is
 * what makes this cover the legacy `widgetStates.gallery` shape for free, and it
 * is safe because these keys occur nowhere else in a project document.
 */
export const stripInstallationState = (projectDocument: Record<string, unknown>): Record<string, unknown> =>
  stripNode(projectDocument) as Record<string, unknown>;

export interface ProjectAssetMappings {
  images: ReadonlyMap<string, string>;
  videos: ReadonlyMap<string, string>;
}

const remapNode = (node: unknown, mappings: ProjectAssetMappings): unknown => {
  if (Array.isArray(node)) {
    let hasChanged = false;
    const next = node.map((item) => {
      const mapped = remapNode(item, mappings);

      hasChanged ||= mapped !== item;

      return mapped;
    });

    return hasChanged ? next : node;
  }

  if (!isRecord(node)) {
    return node;
  }

  let hasChanged = false;
  const next: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(node)) {
    if (typeof value === 'string') {
      const mapping = IMAGE_NAME_KEYS.has(key) ? mappings.images : VIDEO_NAME_KEYS.has(key) ? mappings.videos : null;

      if (mapping) {
        const replacement = mapping.get(value);

        next[key] = replacement ?? value;
        hasChanged ||= replacement !== undefined && replacement !== value;
        continue;
      }
    }

    const mapped = remapNode(value, mappings);

    next[key] = mapped;
    hasChanged ||= mapped !== value;
  }

  return hasChanged ? next : node;
};

/**
 * Rewrite every asset reference in the document through the mapping for its
 * kind. Names absent from their mapping are left alone, and subtrees with
 * nothing to rewrite keep their identity, so empty mappings return the document
 * unchanged.
 */
export const remapAssetRefs = (
  projectDocument: Record<string, unknown>,
  mappings: ProjectAssetMappings
): Record<string, unknown> =>
  mappings.images.size === 0 && mappings.videos.size === 0
    ? projectDocument
    : (remapNode(projectDocument, mappings) as Record<string, unknown>);

const readGalleryRecentImageName = (projectDocument: Record<string, unknown>): string | null => {
  const instances = projectDocument.widgetInstances;

  if (!isRecord(instances)) {
    return null;
  }

  for (const instance of Object.values(instances)) {
    if (!isRecord(instance) || instance.typeId !== 'gallery' || !isRecord(instance.state)) {
      continue;
    }

    const values = instance.state.values;

    if (!isRecord(values) || !Array.isArray(values.recentImages)) {
      continue;
    }

    // `recentImages` is persisted newest-first and bounded (see
    // `getBoundedRecentImages`), so index 0 is the project's latest result.
    for (const image of values.recentImages) {
      if (isRecord(image) && typeof image.imageName === 'string' && image.imageName !== '') {
        return image.imageName;
      }
    }
  }

  return null;
};

const readTopmostCanvasImageName = (projectDocument: Record<string, unknown>): string | null => {
  const canvas = projectDocument.canvas;

  if (!isRecord(canvas) || !isRecord(canvas.document) || !Array.isArray(canvas.document.layers)) {
    return null;
  }

  // Layer index 0 is the top-most layer, which is the one a person would call
  // "what this project looks like".
  for (const layer of canvas.document.layers) {
    if (!isRecord(layer) || !isRecord(layer.source)) {
      continue;
    }

    const source = layer.source;
    const ref = source.type === 'image' ? source.image : source.type === 'paint' ? source.bitmap : null;

    if (isRecord(ref) && typeof ref.imageName === 'string' && ref.imageName !== '') {
      return ref.imageName;
    }
  }

  return null;
};

/**
 * The image that best stands for this project: its newest generated result,
 * falling back to the top-most canvas layer with pixels. `null` for a project
 * that has neither, which is the permanent state for one that has produced
 * nothing — `ProjectCover` renders a glyph for exactly that case.
 */
export const selectCoverImageName = (projectDocument: Record<string, unknown>): string | null =>
  readGalleryRecentImageName(projectDocument) ?? readTopmostCanvasImageName(projectDocument);
