import { deepClone } from 'common/util/deepClone';
import { satisfies } from 'compare-versions';
import { defaultsDeep, keys, pick } from 'es-toolkit/compat';
import { NodeUpdateError } from 'features/nodes/types/error';
import type { InvocationNode, InvocationNodeData, InvocationTemplate } from 'features/nodes/types/invocation';
import { zParsedSemver } from 'features/nodes/types/semver';

import { buildInvocationNode } from './buildInvocationNode';

type UpdateNodeOptions = {
  connectedInputNames?: Set<string>;
};

export const getNeedsUpdate = (data: InvocationNodeData, template: InvocationTemplate): boolean => {
  if (data.type !== template.type) {
    return true;
  }
  return data.version !== template.version;
};

/**
 * Checks if a node may be updated by comparing its major version with the template's major version.
 * @param node The node to check.
 * @param template The invocation template to check against.
 */
const getMayUpdateNode = (node: InvocationNode, template: InvocationTemplate): boolean => {
  const needsUpdate = getNeedsUpdate(node.data, template);
  if (!needsUpdate || node.data.type !== template.type) {
    return false;
  }
  const templateMajor = zParsedSemver.parse(template.version).major;

  return satisfies(node.data.version, `^${templateMajor}`);
};

const migrateImageCollectionInputValues = (node: InvocationNode, options?: UpdateNodeOptions) => {
  if (node.data.type !== 'image_collection') {
    return;
  }

  const collection = node.data.inputs.collection;
  const images = node.data.inputs.images;
  if (!collection || !images || !Array.isArray(collection.value)) {
    return;
  }
  if (Array.isArray(images.value) && images.value.length > 0) {
    return;
  }

  if (!options?.connectedInputNames?.has('collection')) {
    images.value = collection.value;
  }
  collection.value = [];
};

/**
 * Updates a node to the latest version of its template:
 * - Create a new node data object with the latest version of the template.
 * - Recursively merge new node data object into the node to be updated.
 *
 * The input node is not mutated; a new object is returned.
 *
 * @param node The node to be updated.
 * @param template The invocation template to update to.
 * @throws {NodeUpdateError} If the node is not an invocation node.
 */
export const updateNode = (
  node: InvocationNode,
  template: InvocationTemplate,
  options?: UpdateNodeOptions
): InvocationNode => {
  const mayUpdate = getMayUpdateNode(node, template);

  if (!mayUpdate || node.data.type !== template.type) {
    throw new NodeUpdateError(`Unable to update node ${node.id}`);
  }

  // Start with a "fresh" node - just as if the user created a new node of this type
  const defaults = buildInvocationNode(node.position, template);

  // The updateability of a node, via semver comparison, relies on the this kind of recursive merge
  // being valid. We rely on the template's major version to be majorly incremented if this kind of
  // merge would result in an invalid node.
  const clone = deepClone(node);
  clone.data.version = template.version;
  defaultsDeep(clone, defaults); // mutates!
  migrateImageCollectionInputValues(clone, options);

  // Remove any fields that are not in the template
  clone.data.inputs = pick(clone.data.inputs, keys(defaults.data.inputs));
  return clone;
};
