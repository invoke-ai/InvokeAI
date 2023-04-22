# Node Editor Design

WIP

nodes

everything in `src/features/nodes/`

have a look at `state.nodes.invocation`

- on socket connect, if no schema saved, fetch `localhost:9090/openapi.json`, save JSON to `state.nodes.schema`
- on fulfilled schema fetch, `parseSchema()` the schema. this outputs a `Record<string, Invocation>` which is saved to `state.nodes.invocations` - `Invocation` is like a template for the node
- when you add a node, the the `Invocation` template is passed to `InvocationComponent.tsx` to build the UI component for that node
- inputs/outputs have field types - and each field type gets an `FieldComponent` which includes a dispatcher to write state changes to redux `nodesSlice`
- `reactflow` sends changes to nodes/edges to redux
- to invoke, `buildNodesGraph()` state, then send this
- changed onClick Invoke button actions to build the schema, then when schema builds it dispatches the actual network request to create the session - see `session.ts`
