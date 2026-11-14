/**
 * Every supported video base, variant and generation mode compiles a graph — the systematic
 * counterpart to `graph.test.ts`, and the video twin of
 * `src/features/generation/core/graphCoverage.test.ts`.
 *
 * `graph.test.ts` asserts what an individual family wires, one hand-written case at a time. A
 * variant added to `VIDEO_GENERATION` but never given a case is silently untested, and so is a mode
 * a variant declares but no case exercises. This file iterates the registry itself, so a new
 * architecture is covered the moment it is registered.
 *
 * It also records the node types, edge fields and literal input values each case emits into a
 * committed fixture. That fixture is the frontend half of a cross-stack contract:
 * `tests/app/invocations/test_frontend_graph_node_types.py` reads it, alongside the Generate
 * panel's own, and checks every type and field against the backend's `InvocationRegistry`. The
 * frontend never sees a backend schema,
 * so a node renamed or a field moved is otherwise an enqueue-time error on a graph the user cannot
 * edit — after the model has loaded.
 *
 * The snapshot file is written by this test and by nothing else; `.oxfmtrc.json` excludes it so
 * the formatter and this test are not both in charge of the same bytes.
 */

import { describe, expect, it } from 'vitest';

import { AUTO_DURATION_CASES, cases, compileForCase } from './videoGraphCoverage.testing';
import { SUPPORTED_VIDEO_BASES, VIDEO_GENERATION } from './videoPolicies';

describe('video graph coverage', () => {
  it('registers a variant set for every supported base', () => {
    for (const base of SUPPORTED_VIDEO_BASES) {
      expect(Object.keys(VIDEO_GENERATION[base].variants).length, `${base} registers no variants`).toBeGreaterThan(0);
    }
  });

  it('compiles at least one case for every registered variant', () => {
    // The selectability filter in `cases` drops shapes the panel refuses; a variant it drops entirely
    // would leave this suite silently covering nothing for it.
    const covered = new Set(cases.map((testCase) => `${testCase.base}/${testCase.variant}`));
    const registered = SUPPORTED_VIDEO_BASES.flatMap((base) =>
      Object.keys(VIDEO_GENERATION[base].variants).map((variant) => `${base}/${variant}`)
    );

    expect([...covered].sort()).toEqual(registered.sort());
  });

  it.each(cases)('compiles a structurally sound graph for $label', (testCase) => {
    const { graph } = compileForCase(testCase);
    const nodeIds = new Set(Object.keys(graph.nodes));

    expect(nodeIds.size).toBeGreaterThan(0);

    for (const [id, node] of Object.entries(graph.nodes)) {
      expect(node.id, `node keyed '${id}' carries a mismatched id`).toBe(id);
      expect(node.type, `node '${id}' has no type`).toBeTruthy();
    }

    for (const edge of graph.edges) {
      const description = `${edge.source.node_id}.${edge.source.field} -> ${edge.destination.node_id}.${edge.destination.field}`;

      expect(nodeIds.has(edge.source.node_id), `dangling source in edge ${description}`).toBe(true);
      expect(nodeIds.has(edge.destination.node_id), `dangling destination in edge ${description}`).toBe(true);
      expect(edge.source.field, `edge ${description} has no source field`).toBeTruthy();
      expect(edge.destination.field, `edge ${description} has no destination field`).toBeTruthy();
    }
  });

  it('emits the node types and fields the backend has to provide', async () => {
    const byBase: Record<string, { modes: string[]; nodeTypes: string[] }> = {};
    const fields: Record<
      string,
      { inputs: Set<string>; literals: Map<string, Map<string, unknown>>; outputs: Set<string> }
    > = {};

    const connections = new Set<string>();
    const fieldsFor = (nodeType: string) =>
      (fields[nodeType] ??= { inputs: new Set(), literals: new Map(), outputs: new Set() });

    for (const testCase of [...cases, ...AUTO_DURATION_CASES]) {
      const { graph } = compileForCase(testCase);
      const entry = (byBase[testCase.base] ??= { modes: [], nodeTypes: [] });

      entry.modes = [...new Set([...entry.modes, testCase.mode])].sort();
      entry.nodeTypes = [
        ...new Set([...entry.nodeTypes, ...Object.values(graph.nodes).map((node) => node.type)]),
      ].sort();

      for (const node of Object.values(graph.nodes)) {
        const { literals } = fieldsFor(node.type);

        for (const [field, value] of Object.entries(node)) {
          // `id` and `type` address the node rather than feed it, and an undefined value is not
          // serialized into the request at all — the builders use it to mean "leave the default".
          if (field === 'id' || field === 'type' || value === undefined) {
            continue;
          }

          const values = literals.get(field) ?? new Map<string, unknown>();
          literals.set(field, values);

          // Scalars only: the object-valued inputs are model identifiers and media fields built
          // from this file's synthetic fixtures, so validating them would assert something about
          // the fixture rather than about the contract.
          if (value === null || typeof value !== 'object') {
            values.set(JSON.stringify(value), value);
          }
        }
      }

      for (const edge of graph.edges) {
        fieldsFor(graph.nodes[edge.source.node_id]!.type).outputs.add(edge.source.field);
        fieldsFor(graph.nodes[edge.destination.node_id]!.type).inputs.add(edge.destination.field);
        // The connection itself, by node TYPE, so the Python side can ask the backend whether the
        // two field types are actually compatible. Field names alone are not enough: a float wired
        // into an int is two valid names and a graph the queue refuses at enqueue.
        connections.add(
          [
            graph.nodes[edge.source.node_id]!.type,
            edge.source.field,
            graph.nodes[edge.destination.node_id]!.type,
            edge.destination.field,
          ].join(' ')
        );
      }
    }

    const contract = {
      connections: [...connections].sort(),
      _comment:
        'Generated by src/features/video/core/videoGraphCoverage.test.ts; regenerate with ' +
        '`vitest -u`. Excluded from oxfmt so the test alone owns its layout. Consumed by ' +
        'tests/app/invocations/test_frontend_graph_node_types.py, which checks every node type, ' +
        'edge field and literal input value below against the backend invocation registry, and ' +
        'every connection for whether the backend considers the two field types compatible. An ' +
        'empty literalInputs value list means the field is set to something other than a scalar, ' +
        'so only its name is checked.',
      byBase,
      fieldsByNodeType: Object.fromEntries(
        Object.entries(fields)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([nodeType, { inputs, literals, outputs }]) => [
            nodeType,
            {
              inputs: [...inputs].sort(),
              literalInputs: Object.fromEntries(
                [...literals]
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([field, values]) => [
                    field,
                    [...values].sort(([a], [b]) => a.localeCompare(b)).map(([, value]) => value),
                  ])
              ),
              outputs: [...outputs].sort(),
            },
          ])
      ),
    };

    await expect(`${JSON.stringify(contract, null, 2)}\n`).toMatchFileSnapshot(
      './__snapshots__/videoGraphNodeTypes.json'
    );
  });
});
