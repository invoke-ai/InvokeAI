import type {
  CanvasDocumentContractV3,
  CanvasNodeContract,
  CanvasStateContractV3,
  RegionalGuidanceReferenceImage,
} from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';

import { GALLERY_ITEM_URL_FIELDS, GENERATED_IMAGE_URL_FIELDS } from './projectAssets';

/** Builds one kind of media's URLs from its name, against the deployment the page runs on. */
export interface MediaUrlBuilders {
  readonly full: (name: string) => string;
  readonly thumbnail: (name: string) => string;
}

export interface ProjectMediaUrlResolver {
  readonly image: MediaUrlBuilders;
  readonly video: MediaUrlBuilders;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isName = (value: unknown): value is string => typeof value === 'string' && value.length > 0;

type UrlFields = Readonly<Record<string, keyof MediaUrlBuilders>>;

/** The value with `fields` derived from `name`; the value itself when every URL already matches, keeping identity. */
const withUrls = <T extends Record<string, unknown>>(
  value: T,
  fields: UrlFields,
  builders: MediaUrlBuilders,
  name: string
): T => {
  let next: Record<string, unknown> | null = null;
  for (const [key, builder] of Object.entries(fields)) {
    const url = builders[builder](name);
    if (value[key] !== url) {
      next ??= { ...value };
      next[key] = url;
    }
  }
  return (next as T | null) ?? value;
};

const mapShared = <T>(items: T[], rebuild: (item: T) => T): T[] => {
  let next: T[] | null = null;
  items.forEach((item, index) => {
    const rebuilt = rebuild(item);
    if (rebuilt !== item) {
      next ??= items.slice();
      next[index] = rebuilt;
    }
  });
  return next ?? items;
};

/** A generated-image record (`imageName` beside `imageUrl`/`thumbnailUrl`): recents, staging, references. */
const rebuildGeneratedImage = <T>(value: T, urls: ProjectMediaUrlResolver): T =>
  isRecord(value) && isName(value.imageName)
    ? (withUrls(value, GENERATED_IMAGE_URL_FIELDS, urls.image, value.imageName) as T)
    : value;

/** A gallery selection: a gallery item (`name` and `kind`), or the generated-image shape older documents stored. */
const rebuildGallerySelection = (value: unknown, urls: ProjectMediaUrlResolver): unknown => {
  if (!isRecord(value)) {
    return value;
  }
  if (isName(value.name) && (value.kind === 'image' || value.kind === 'video')) {
    return withUrls(value, GALLERY_ITEM_URL_FIELDS, value.kind === 'video' ? urls.video : urls.image, value.name);
  }
  return rebuildGeneratedImage(value, urls);
};

const rebuildGalleryValues = (
  values: Record<string, unknown>,
  urls: ProjectMediaUrlResolver
): Record<string, unknown> => {
  const selectedImage = rebuildGallerySelection(values.selectedImage, urls);
  const compareImage = rebuildGallerySelection(values.compareImage, urls);
  const recentImages = Array.isArray(values.recentImages)
    ? mapShared(values.recentImages as unknown[], (image) => rebuildGeneratedImage(image, urls))
    : values.recentImages;
  return selectedImage === values.selectedImage &&
    compareImage === values.compareImage &&
    recentImages === values.recentImages
    ? values
    : { ...values, compareImage, recentImages, selectedImage };
};

const rebuildWidgetInstances = (
  instances: Project['widgetInstances'],
  urls: ProjectMediaUrlResolver
): Project['widgetInstances'] => {
  let next: Project['widgetInstances'] | null = null;
  for (const [instanceId, instance] of Object.entries(instances)) {
    if (instance.typeId !== 'gallery') {
      continue;
    }
    const values = rebuildGalleryValues(instance.state.values, urls);
    if (values !== instance.state.values) {
      next ??= { ...instances };
      next[instanceId] = { ...instance, state: { ...instance.state, values } };
    }
  }
  return next ?? instances;
};

const rebuildReferenceImage = (
  reference: RegionalGuidanceReferenceImage,
  urls: ProjectMediaUrlResolver
): RegionalGuidanceReferenceImage => {
  const image = rebuildGeneratedImage(reference.config.image, urls);
  return image === reference.config.image ? reference : { ...reference, config: { ...reference.config, image } };
};

const rebuildRegionalNode = (node: CanvasNodeContract, urls: ProjectMediaUrlResolver): CanvasNodeContract => {
  if (node.type === 'group') {
    const children = mapShared(node.children, (child) => rebuildRegionalNode(child, urls));
    return children === node.children ? node : { ...node, children };
  }
  if (node.type !== 'regional_guidance') {
    return node;
  }
  const referenceImages = mapShared(node.referenceImages, (reference) => rebuildReferenceImage(reference, urls));
  return referenceImages === node.referenceImages ? node : { ...node, referenceImages };
};

/** Regional guidance is the only canvas layer that keeps image URLs; every other leaf holds names alone. */
const rebuildCanvasDocument = (
  document: CanvasDocumentContractV3,
  urls: ProjectMediaUrlResolver
): CanvasDocumentContractV3 => {
  const regional = document.stacks.regional_guidance;
  const rebuilt = mapShared(regional, (node) => rebuildRegionalNode(node, urls));
  return rebuilt === regional ? document : { ...document, stacks: { ...document.stacks, regional_guidance: rebuilt } };
};

const rebuildCanvas = (canvas: CanvasStateContractV3, urls: ProjectMediaUrlResolver): CanvasStateContractV3 => {
  const document = rebuildCanvasDocument(canvas.document, urls);
  const snapshots = mapShared(canvas.snapshots, (snapshot) => {
    const snapshotDocument = rebuildCanvasDocument(snapshot.document, urls);
    return snapshotDocument === snapshot.document ? snapshot : { ...snapshot, document: snapshotDocument };
  });
  const pendingImages = mapShared(canvas.stagingArea.pendingImages, (image) => rebuildGeneratedImage(image, urls));
  return document === canvas.document &&
    snapshots === canvas.snapshots &&
    pendingImages === canvas.stagingArea.pendingImages
    ? canvas
    : {
        ...canvas,
        document,
        snapshots,
        stagingArea:
          pendingImages === canvas.stagingArea.pendingImages
            ? canvas.stagingArea
            : { ...canvas.stagingArea, pendingImages },
      };
};

/**
 * Rebuilds every media URL a project stores from the name stored beside it. Stored URLs name the origin that saved
 * them (`http://localhost:9090/...`), which another origin or machine cannot reach, and `.invk` transfer blanks them,
 * so each load derives them again for the deployment the page runs on. Names are the durable reference; a URL is
 * overwritten whatever it held, and an item without a name is left alone. Unchanged subtrees keep their identity.
 */
export const rebuildProjectMediaUrls = (project: Project, urls: ProjectMediaUrlResolver): Project => {
  const widgetInstances = rebuildWidgetInstances(project.widgetInstances, urls);
  const canvas = rebuildCanvas(project.canvas, urls);
  return widgetInstances === project.widgetInstances && canvas === project.canvas
    ? project
    : { ...project, canvas, widgetInstances };
};
