import type { NodeSettingName } from 'features/nodes/types/workflow';

/**
 * Key a node setting override is stored under in `canvasWorkflowIntegrationSlice.nodeSettingValues`.
 *
 * Deliberately shaped like the `fieldValues` key (`"<nodeId>.<name>"`) but kept in its own record: node settings are
 * not input fields, so they must not be written into `node.data.inputs` the way field values are.
 */
export const getNodeSettingKey = (nodeId: string, setting: NodeSettingName): string => `${nodeId}.${setting}`;

type ResolveNodeSettingsArgs = {
  /** The node's id as saved in the workflow, i.e. before canvas output ids are re-prefixed. */
  nodeId: string;
  isCanvasOutputNode: boolean;
  /** `node.data.isIntermediate` as saved in the workflow. */
  isIntermediate: boolean | undefined;
  /** `node.data.useCache` as saved in the workflow. */
  useCache: boolean | undefined;
  nodeSettingValues: Record<string, boolean> | null | undefined;
  isAdmin: boolean;
};

/**
 * Resolve the `use_cache` / `is_intermediate` an enqueued invocation runs with, given what the workflow was saved
 * with and what the user changed in the canvas form preview.
 *
 * Without this the preview's node settings would be inert - the graph is built from node data, which the preview does
 * not own - so a form's Use Cache or Save To Gallery control would silently do nothing.
 */
export const resolveNodeSettings = ({
  nodeId,
  isCanvasOutputNode,
  isIntermediate,
  useCache,
  nodeSettingValues,
  isAdmin,
}: ResolveNodeSettingsArgs): { use_cache: boolean; is_intermediate: boolean } => {
  const saveToGalleryOverride = nodeSettingValues?.[getNodeSettingKey(nodeId, 'save_to_gallery')];
  // The node cache is a process-global, admin-only policy. The slice is persisted per browser, not per user, so a
  // value left behind by an admin must not be replayed for whoever runs the workflow next.
  const useCacheOverride = isAdmin ? nodeSettingValues?.[getNodeSettingKey(nodeId, 'use_cache')] : undefined;

  return {
    // Canvas output images go to the staging area, never straight to the gallery, so this one is not the user's to
    // set. The preview does not offer a control for it either.
    is_intermediate: isCanvasOutputNode
      ? true
      : saveToGalleryOverride !== undefined
        ? !saveToGalleryOverride
        : (isIntermediate ?? false),
    use_cache: useCacheOverride ?? useCache ?? true,
  };
};
