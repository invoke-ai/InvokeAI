import { type ModelIdentifierField, zModelIdentifierField } from 'features/nodes/types/common';
import { METADATA } from 'features/nodes/util/graph/constants';
import { forEach, groupBy, isEqual, unset, values } from 'lodash-es';
import type {
  AnyInvocation,
  AnyInvocationIncMetadata,
  AnyInvocationInputField,
  AnyInvocationOutputField,
  AnyModelConfig,
  CoreMetadataInvocation,
  InputFields,
  Invocation,
  InvocationType,
  OutputFields,
} from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

type Edge = {
  source: {
    node_id: string;
    field: AnyInvocationOutputField;
  };
  destination: {
    node_id: string;
    field: AnyInvocationInputField;
  };
};

export type GraphType = { id: string; nodes: Record<string, AnyInvocation>; edges: Edge[] };

export class Graph {
  _graph: GraphType;

  constructor(id?: string) {
    this._graph = {
      id: id ?? Graph.uuid(),
      nodes: {},
      edges: [],
    };
  }

  //#region Node Operations

  /**
   * Add a node to the graph. If a node with the same id already exists, an `AssertionError` is raised.
   * The optional `is_intermediate` and `use_cache` fields are both set to `true`, if not set on the node.
   * @param node The node to add.
   * @returns The added node.
   * @raises `AssertionError` if a node with the same id already exists.
   */
  addNode<T extends InvocationType>(node: Invocation<T>): Invocation<T> {
    assert(this._graph.nodes[node.id] === undefined, Graph.getNodeAlreadyExistsMsg(node.id));
    if (node.is_intermediate === undefined) {
      node.is_intermediate = true;
    }
    if (node.use_cache === undefined) {
      node.use_cache = true;
    }
    this._graph.nodes[node.id] = node;
    return node;
  }

  /**
   * Gets a node from the graph.
   * @param id The id of the node to get.
   * @returns The node.
   * @raises `AssertionError` if the node does not exist.
   */
  getNode(id: string): AnyInvocation {
    const node = this._graph.nodes[id];
    assert(node !== undefined, Graph.getNodeNotFoundMsg(id));
    return node;
  }

  /**
   * Deletes a node from the graph. All edges to and from the node are also deleted.
   * @param id The id of the node to delete.
   */
  deleteNode(id: string): void {
    const node = this._graph.nodes[id];
    if (node) {
      this.deleteEdgesFrom(node);
      this.deleteEdgesTo(node);
      delete this._graph.nodes[id];
    }
  }

  /**
   * Check if a node exists in the graph.
   * @param id The id of the node to check.
   * @returns Whether the graph has a node with the given id.
   */
  hasNode(id: string): boolean {
    try {
      this.getNode(id);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get the immediate incomers of a node.
   * @param node The node to get the incomers of.
   * @returns The incoming nodes.
   * @raises `AssertionError` if one of the target node's incoming edges has an invalid source node.
   */
  getIncomers(node: AnyInvocation): AnyInvocation[] {
    return this.getEdgesTo(node).map((edge) => this.getNode(edge.source.node_id));
  }

  /**
   * Get the immediate outgoers of a node.
   * @param node The node to get the outgoers of.
   * @returns The outgoing nodes.
   * @raises `AssertionError` if one of the target node's outgoing edges has an invalid destination node.
   */
  getOutgoers(node: AnyInvocation): AnyInvocation[] {
    return this.getEdgesFrom(node).map((edge) => this.getNode(edge.destination.node_id));
  }
  //#endregion

  //#region Edge Operations

  /**
   * Add an edge to the graph. If an edge with the same source and destination already exists, an `AssertionError` is raised.
   * @param fromNode The source node.
   * @param fromField The field of the source node.
   * @param toNode The destination node.
   * @param toField The field of the destination node.
   * @returns The added edge.
   * @raises `AssertionError` if an edge with the same source and destination already exists.
   */
  addEdge<TFrom extends AnyInvocation, TTo extends AnyInvocation>(
    fromNode: TFrom,
    fromField: OutputFields<TFrom>,
    toNode: TTo,
    toField: InputFields<TTo>
  ): Edge {
    const edge: Edge = {
      source: { node_id: fromNode.id, field: fromField },
      destination: { node_id: toNode.id, field: toField },
    };
    const edgeAlreadyExists = this._graph.edges.some((e) => isEqual(e, edge));
    assert(!edgeAlreadyExists, Graph.getEdgeAlreadyExistsMsg(fromNode.id, fromField, toNode.id, toField));
    this._graph.edges.push(edge);
    return edge;
  }

  /**
   * Add an edge to the graph. If an edge with the same source and destination already exists, an `AssertionError` is raised.
   * @param edge The edge to add.
   * @returns The added edge.
   * @raises `AssertionError` if an edge with the same source and destination already exists.
   */
  addEdgeFromObj(edge: Edge): Edge {
    const edgeAlreadyExists = this._graph.edges.some((e) => isEqual(e, edge));
    assert(
      !edgeAlreadyExists,
      Graph.getEdgeAlreadyExistsMsg(
        edge.source.node_id,
        edge.source.field,
        edge.destination.node_id,
        edge.destination.field
      )
    );
    this._graph.edges.push(edge);
    return edge;
  }

  /**
   * Get an edge from the graph. If the edge does not exist, an `AssertionError` is raised.
   * @param fromNode The source node.
   * @param fromField The field of the source node.
   * @param toNode The destination node.
   * @param toField The field of the destination node.
   * @returns The edge.
   * @raises `AssertionError` if the edge does not exist.
   */
  getEdge<TFrom extends AnyInvocation, TTo extends AnyInvocation>(
    fromNode: TFrom,
    fromField: OutputFields<TFrom>,
    toNode: TTo,
    toField: InputFields<TTo>
  ): Edge {
    const edge = this._graph.edges.find(
      (e) =>
        e.source.node_id === fromNode.id &&
        e.source.field === fromField &&
        e.destination.node_id === toNode.id &&
        e.destination.field === toField
    );
    assert(edge !== undefined, Graph.getEdgeNotFoundMsg(fromNode.id, fromField, toNode.id, toField));
    return edge;
  }

  /**
   * Get all edges in the graph.
   * @returns The edges.
   */
  getEdges(): Edge[] {
    return this._graph.edges;
  }

  /**
   * Check if a graph has an edge.
   * @param fromNode The source node.
   * @param fromField The field of the source node.
   * @param toNode The destination node.
   * @param toField The field of the destination node.
   * @returns Whether the graph has the edge.
   */

  hasEdge<TFrom extends AnyInvocation, TTo extends AnyInvocation>(
    fromNode: TFrom,
    fromField: OutputFields<TFrom>,
    toNode: TTo,
    toField: InputFields<TTo>
  ): boolean {
    try {
      this.getEdge(fromNode, fromField, toNode, toField);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get all edges from a node. If `fromFields` is provided, only edges from those fields are returned.
   * @param fromNode The source node.
   * @param fromFields The fields of the source node (optional).
   * @returns The edges.
   */
  getEdgesFrom<T extends AnyInvocation>(fromNode: T, fromFields?: OutputFields<T>[]): Edge[] {
    let edges = this._graph.edges.filter((edge) => edge.source.node_id === fromNode.id);
    if (fromFields) {
      // TODO(psyche): figure out how to satisfy TS here without casting - this is _not_ an unsafe cast
      edges = edges.filter((edge) => (fromFields as AnyInvocationOutputField[]).includes(edge.source.field));
    }
    return edges;
  }

  /**
   * Get all edges to a node. If `toFields` is provided, only edges to those fields are returned.
   * @param toNodeId The destination node.
   * @param toFields The fields of the destination node (optional).
   * @returns The edges.
   */
  getEdgesTo<T extends AnyInvocation>(toNode: T, toFields?: InputFields<T>[]): Edge[] {
    let edges = this._graph.edges.filter((edge) => edge.destination.node_id === toNode.id);
    if (toFields) {
      edges = edges.filter((edge) => (toFields as AnyInvocationInputField[]).includes(edge.destination.field));
    }
    return edges;
  }

  /**
   * INTERNAL: Delete _all_ matching edges from the graph. Uses _.isEqual for comparison.
   * @param edge The edge to delete
   */
  private _deleteEdge(edge: Edge): void {
    this._graph.edges = this._graph.edges.filter((e) => !isEqual(e, edge));
  }

  /**
   * Delete all edges to a node. If `toFields` is provided, only edges to those fields are deleted.
   * @param toNode The destination node.
   * @param toFields The fields of the destination node (optional).
   */
  deleteEdgesTo<T extends AnyInvocation>(toNode: T, toFields?: InputFields<T>[]): void {
    for (const edge of this.getEdgesTo(toNode, toFields)) {
      this._deleteEdge(edge);
    }
  }

  /**
   * Delete all edges from a node. If `fromFields` is provided, only edges from those fields are deleted.
   * @param toNode The id of the source node.
   * @param fromFields The fields of the source node (optional).
   */
  deleteEdgesFrom<T extends AnyInvocation>(fromNode: T, fromFields?: OutputFields<T>[]): void {
    for (const edge of this.getEdgesFrom(fromNode, fromFields)) {
      this._deleteEdge(edge);
    }
  }
  //#endregion

  //#region Graph Ops

  /**
   * Validate the graph. Checks that all edges have valid source and destination nodes.
   * @raises `AssertionError` if an edge has an invalid source or destination node.
   */
  validate(): void {
    // TODO(psyche): Add more validation checks - cycles, valid invocation types, etc.
    for (const edge of this._graph.edges) {
      this.getNode(edge.source.node_id);
      this.getNode(edge.destination.node_id);
      assert(
        !this._graph.edges.filter((e) => e !== edge).find((e) => isEqual(e, edge)),
        `Duplicate edge: ${Graph.edgeToString(edge)}`
      );
    }
    for (const node of values(this._graph.nodes)) {
      const edgesTo = this.getEdgesTo(node);
      // Validate that no node has multiple incoming edges with the same field
      forEach(groupBy(edgesTo, 'destination.field'), (group, field) => {
        if (node.type === 'collect' && field === 'item') {
          // Collectors' item field accepts multiple incoming edges
          return;
        }
        assert(
          group.length === 1,
          `Node ${node.id} has multiple incoming edges with field ${field}: ${group.map(Graph.edgeToString).join(', ')}`
        );
      });
    }
  }

  /**
   * Gets the graph after validating it.
   * @returns The graph.
   * @raises `AssertionError` if the graph is invalid.
   */
  getGraph(): GraphType {
    this.validate();
    return this._graph;
  }

  /**
   * Gets the graph without validating it.
   * @returns The graph.
   */
  getGraphSafe(): GraphType {
    return this._graph;
  }
  //#endregion

  //#region Metadata

  /**
   * INTERNAL: Get the metadata node. If it does not exist, it is created.
   * @returns The metadata node.
   */
  _getMetadataNode(): CoreMetadataInvocation {
    try {
      const node = this.getNode(METADATA) as AnyInvocationIncMetadata;
      assert(node.type === 'core_metadata');
      return node;
    } catch {
      const node: CoreMetadataInvocation = { id: METADATA, type: 'core_metadata' };
      // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
      return this.addNode(node);
    }
  }

  /**
   * Add metadata to the graph. If the metadata node does not exist, it is created. If the specific metadata key exists,
   * it is overwritten.
   * @param metadata The metadata to add.
   * @returns The metadata node.
   */
  upsertMetadata(metadata: Partial<CoreMetadataInvocation>): CoreMetadataInvocation {
    const node = this._getMetadataNode();
    Object.assign(node, metadata);
    return node;
  }

  /**
   * Remove metadata from the graph.
   * @param keys The keys of the metadata to remove
   * @returns The metadata node
   */
  removeMetadata(keys: string[]): CoreMetadataInvocation {
    const metadataNode = this._getMetadataNode();
    for (const k of keys) {
      unset(metadataNode, k);
    }
    return metadataNode;
  }

  /**
   * Set the node that should receive metadata. All other edges from the metadata node are deleted.
   * @param node The node to set as the receiving node
   */
  setMetadataReceivingNode(node: AnyInvocation): void {
    // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
    this.deleteEdgesFrom(this._getMetadataNode());
    // @ts-expect-error `Graph` excludes `core_metadata` nodes due to its excessively wide typing
    this.addEdge(this._getMetadataNode(), 'metadata', node, 'metadata');
  }

  /**
   * Given a model config, return the model metadata field.
   * @param modelConfig The model config entity
   * @returns
   */
  static getModelMetadataField(modelConfig: AnyModelConfig): ModelIdentifierField {
    return zModelIdentifierField.parse(modelConfig);
  }
  //#endregion

  //#region Util
  static getNodeNotFoundMsg(id: string): string {
    return `Node ${id} not found`;
  }

  static getNodeNotOfTypeMsg(node: AnyInvocation, expectedType: InvocationType): string {
    return `Node ${node.id} is not of type ${expectedType}: ${node.type}`;
  }

  static getNodeAlreadyExistsMsg(id: string): string {
    return `Node ${id} already exists`;
  }

  static getEdgeNotFoundMsg(fromNodeId: string, fromField: string, toNodeId: string, toField: string) {
    return `Edge from ${fromNodeId}.${fromField} to ${toNodeId}.${toField} not found`;
  }

  static getEdgeAlreadyExistsMsg(fromNodeId: string, fromField: string, toNodeId: string, toField: string) {
    return `Edge from ${fromNodeId}.${fromField} to ${toNodeId}.${toField} already exists`;
  }

  static edgeToString(edge: Edge): string {
    return `${edge.source.node_id}.${edge.source.field} -> ${edge.destination.node_id}.${edge.destination.field}`;
  }

  static uuid = uuidv4;
  //#endregion
}
