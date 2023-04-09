# Node Editor Design

WIP

- on socket connect, if no schema loaded, fetch `localhost:9090/openapi.json`, saved to `state.nodes.schema`
- on fulfilled fetch, `parseSchema()` the schema. this outputs a `Record<string, Invocation>` which is saved to `state.nodes.invocations`
- when you add a node, it gives it to `InvocationComponent.tsx`