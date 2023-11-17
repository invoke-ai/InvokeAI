import { satisfies } from 'compare-versions';
import { NodeUpdateError } from 'features/nodes/types/error';
import {
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/invocation';
import { zParsedSemver } from 'features/nodes/types/semver';
import { cloneDeep, defaultsDeep } from 'lodash-es';
import { Node } from 'reactflow';
import { buildInvocationNode } from './buildNodeData';

export const getNeedsUpdate = (
  node: Node<InvocationNodeData>,
  template: InvocationTemplate
): boolean => {
  if (node.data.type !== template.type) {
    return true;
  }
  return node.data.version !== template.version;
}; /**
 * Checks if a node may be updated by comparing its major version with the template's major version.
 * @param node The node to check.
 * @param template The invocation template to check against.
 */

export const getMayUpdateNode = (
  node: Node<InvocationNodeData>,
  template: InvocationTemplate
): boolean => {
  const needsUpdate = getNeedsUpdate(node, template);
  if (!needsUpdate || node.data.type !== template.type) {
    return false;
  }
  const templateMajor = zParsedSemver.parse(template.version).major;

  return satisfies(node.data.version, `^${templateMajor}`);
}; /**
 * Updates a node to the latest version of its template:
 * - Create a new node data object with the latest version of the template.
 * - Recursively merge new node data object into the node to be updated.
 *
 * @param node The node to updated.
 * @param template The invocation template to update to.
 * @throws {NodeUpdateError} If the node is not an invocation node.
 */

export const updateNode = (
  node: Node<InvocationNodeData>,
  template: InvocationTemplate
): Node<InvocationNodeData> => {
  const mayUpdate = getMayUpdateNode(node, template);

  if (!mayUpdate || node.data.type !== template.type) {
    throw new NodeUpdateError(`Unable to update node ${node.id}`);
  }

  // Start with a "fresh" node - just as if the user created a new node of this type
  const defaults = buildInvocationNode(node.position, template);

  // The updateability of a node, via semver comparison, relies on the this kind of recursive merge
  // being valid. We rely on the template's major version to be majorly incremented if this kind of
  // merge would result in an invalid node.
  const clone = cloneDeep(node);
  clone.data.version = template.version;
  defaultsDeep(clone, defaults); // mutates!

  return clone;
};
