/**
 * Which assets a project document points at, and how to rewrite those pointers.
 *
 * ### Collection is by key, not by path
 *
 * Three keys hold an asset name: `imageName`, `image_name` and `video_name`. Collecting every
 * string found at those keys is complete by construction — a new adapter kind or node field is
 * picked up without touching this file. The legacy frontend walks a hand-written list of paths
 * instead, which is why it enumerates `ip_adapter` and `flux_redux` by name and drops the pixels of
 * anything added since. A false positive costs nothing: it fails to resolve and is skipped.
 *
 * Images and videos get a set and a mapping each, because they are separate backend namespaces and
 * could share a name — one flat map would let one rewrite the other.
 *
 * ### Collection skips history, remapping does not
 *
 * Queue entries embed whole canvas snapshots and the gallery keeps sixty recents, so bundling
 * everything turns a working project into a multi-gigabyte archive; {@link collectLiveAssetRefs}
 * walks only what the project needs to open. {@link remapAssetRefs} walks everything, because a
 * renamed asset's references must all follow — including the history we chose not to bundle, which
 * would otherwise point at the pre-import name forever.
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
 * The gallery widget's selection: per-install content, so a project arriving elsewhere opens with
 * nothing selected rather than dragging a stranger's gallery behind it.
 *
 * Skipping them during collection is only half of it — an unbundled, unstripped reference still
 * travels, just broken, and the same skip hides it from the restore pass so import cannot even
 * report it as dangling. {@link stripGallerySelection} is the other half.
 */
export const GALLERY_SELECTION_KEYS: ReadonlySet<string> = new Set([
  'compareImage',
  'selectedImage',
  'selectedImageName',
  'selectedImageNames',
]);

/**
 * Board ids the gallery widget holds. Installation state, not project state: `projectBoardId` is a
 * cache the server owns and hydration overwrites.
 *
 * Board ids elsewhere in the document — a workflow node's board input — are meaningful and
 * deliberately NOT stripped, which is why this names only the gallery widget's two keys.
 */
export const GALLERY_INSTALLATION_KEYS: ReadonlySet<string> = new Set(['projectBoardId', 'selectedBoardId']);

/**
 * Where the gallery was scrolled to. Meaningless in another install: the board
 * ids above are stripped, so an imported project lands on a different (usually
 * near-empty) board, where a page number — or, in infinite mode, the mid-board
 * window anchor it doubles as — describes a position that does not exist and
 * renders as an empty board.
 */
export const GALLERY_POSITION_KEYS: ReadonlySet<string> = new Set(['galleryPage']);

const INSTALLATION_STATE_KEYS: ReadonlySet<string> = new Set([
  ...GALLERY_SELECTION_KEYS,
  ...GALLERY_INSTALLATION_KEYS,
  ...GALLERY_POSITION_KEYS,
]);

/**
 * URLs the document caches beside a media name, naming *this* install's copy.
 *
 * Blanked, not removed: `getBoundedRecentImages` requires both to be strings and drops any entry
 * missing one, so deleting them would discard the whole gallery-recents overlay on import. Every
 * consumer already falls back to deriving the URL from the name, which is the only correct answer
 * once a transfer has renamed the media — the name is remapped, a URL built around the old one is
 * not, and would keep resolving to the source project's picture.
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

/** `{ drop: true }` removes the key, `{ value }` replaces it, `null` recurses into it. */
type NodeVisit = { drop: true } | { value: unknown } | null;

/**
 * Walk every entry of a project document at every depth, applying `visit`.
 *
 * Subtrees with nothing to change keep their identity, so a document that matches nothing comes
 * back as the very same object rather than a structurally-equal copy.
 */
const mapDocument = (node: unknown, visit: (key: string, value: unknown) => NodeVisit): unknown => {
  if (Array.isArray(node)) {
    let hasChanged = false;
    const next = node.map((item) => {
      const mapped = mapDocument(item, visit);

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
    const visited = visit(key, value);

    if (visited === null) {
      const mapped = mapDocument(value, visit);

      next[key] = mapped;
      hasChanged ||= mapped !== value;
      continue;
    }

    if ('drop' in visited) {
      hasChanged = true;
      continue;
    }

    next[key] = visited.value;
    hasChanged ||= visited.value !== value;
  }

  return hasChanged ? next : node;
};

/**
 * Drop everything describing *this install* rather than the project, at every depth.
 *
 * Removing a key is the same as clearing it here: every reader already tolerates absence. By key at
 * any depth rather than by walking `widgetInstances`, which covers the legacy `widgetStates.gallery`
 * shape for free and is safe because these keys occur nowhere else in a document.
 */
export const stripInstallationState = (projectDocument: Record<string, unknown>): Record<string, unknown> =>
  mapDocument(projectDocument, (key, value) => {
    if (INSTALLATION_STATE_KEYS.has(key)) {
      return { drop: true };
    }

    return DERIVED_URL_KEYS.has(key) && typeof value === 'string' ? { value: '' } : null;
  }) as Record<string, unknown>;

export interface ProjectAssetMappings {
  images: ReadonlyMap<string, string>;
  videos: ReadonlyMap<string, string>;
}

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
    : (mapDocument(projectDocument, (key, value) => {
        if (typeof value !== 'string') {
          return null;
        }

        const mapping = IMAGE_NAME_KEYS.has(key) ? mappings.images : VIDEO_NAME_KEYS.has(key) ? mappings.videos : null;

        return mapping === null ? null : { value: mapping.get(value) ?? value };
      }) as Record<string, unknown>);

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
