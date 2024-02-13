# Workflows - Design and Implementation

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Workflows - Design and Implementation](#workflows---design-and-implementation)
  - [Design](#design)
    - [Linear UI](#linear-ui)
    - [Workflow Editor](#workflow-editor)
      - [Workflows](#workflows)
        - [Workflow -> reactflow state -> InvokeAI graph](#workflow---reactflow-state---invokeai-graph)
        - [Nodes vs Invocations](#nodes-vs-invocations)
        - [Workflow Linear View](#workflow-linear-view)
      - [OpenAPI Schema](#openapi-schema)
        - [Field Instances and Templates](#field-instances-and-templates)
        - [Stateful vs Stateless Fields](#stateful-vs-stateless-fields)
        - [Collection and Polymorphic Fields](#collection-and-polymorphic-fields)
  - [Implementation](#implementation)
    - [zod Schemas and Types](#zod-schemas-and-types)
    - [OpenAPI Schema Parsing](#openapi-schema-parsing)
      - [Parsing Field Types](#parsing-field-types)
        - [Primitive Types](#primitive-types)
        - [Complex Types](#complex-types)
        - [Collection Types](#collection-types)
        - [Collection or Scalar Types](#collection-or-scalar-types)
        - [Optional Fields](#optional-fields)
      - [Building Field Input Templates](#building-field-input-templates)
      - [Building Field Output Templates](#building-field-output-templates)
    - [Managing reactflow State](#managing-reactflow-state)
      - [Building Nodes and Edges](#building-nodes-and-edges)
      - [Building a Workflow](#building-a-workflow)
      - [Loading a Workflow](#loading-a-workflow)
    - [Workflow Migrations](#workflow-migrations)

<!-- /code_chunk_output -->

> This document describes, at a high level, the design and implementation of workflows in the InvokeAI frontend. There are a substantial number of implementation details not included, but which are hopefully clear from the code.

InvokeAI's backend uses graphs, composed of **nodes** and **edges**, to process data and generate images.

Nodes have any number of **input fields** and **output fields**. Edges connect nodes together via their inputs and outputs. Fields have data types which dictate how they may be connected.

During execution, a nodes' outputs may be passed along to any number of other nodes' inputs.

Workflows are an enriched abstraction over a graph.

## Design

InvokeAI provide two ways to build graphs in the frontend: the [Linear UI](#linear-ui) and [Workflow Editor](#workflow-editor).

To better understand the use case and challenges related to workflows, we will review both of these modes.

### Linear UI

This includes the **Text to Image**, **Image to Image** and **Unified Canvas** tabs.

The user-managed parameters on these tabs are stored as simple objects in the application state. When the user invokes, adding a generation to the queue, we internally build a graph from these parameters.

This logic can be fairly complex due to the range of features available and their interactions. Depending on the parameters selected, the graph may be very different. Building graphs in code can be challenging - you are trying to construct a non-linear structure in a linear context.

The simplest graph building logic is for **Text to Image** with a SD1.5 model: [buildLinearTextToImageGraph.ts]

There are many other graph builders in the same directory for different tabs or base models (e.g. SDXL). Some are pretty hairy.

In the Linear UI, we go straight from **simple application state** to **graph** via these builders.

### Workflow Editor

The Workflow Editor is a visual graph editor, allowing users to draw edges from node to node to construct a graph. This _far_ more approachable way to create complex graphs.

InvokeAI uses the [reactflow] library to power the Workflow Editor. It provides both a graph editor UI and manages its own internal graph state.

#### Workflows

A workflow is a representation of a graph plus additional metadata:

- Name
- Description
- Version
- Notes
- [Exposed fields](#workflow-linear-view)
- Author, tags, category, etc.

Workflows should have other qualities:

- Portable: you should be able to load a workflow created by another person.
- Resilient: you should be able to "upgrade" a workflow as the application changes.
- Abstract: as much as is possible, workflows should not be married to the specific implementation details of the application.

To support these qualities, workflows are serializable, have a versioned schemas, and represent graphs as minimally as possible. Fortunately, the reactflow state for nodes and edges works perfectly for this.

##### Workflow -> reactflow state -> InvokeAI graph

Given a workflow, we need to be able to derive reactflow state and/or an InvokeAI graph from it.

The first step - workflow to reactflow state - is very simple. The logic is in [nodesSlice.ts], in the `workflowLoaded` reducer.

The reactflow state is, however, structurally incompatible with our backend's graph structure. When a user invokes on a Workflow, we need to convert the reactflow state into an InvokeAI graph. This is far simpler than the graph building logic from the Linear UI:
[buildNodesGraph.ts]

##### Nodes vs Invocations

We often use the terms "node" and "invocation" interchangeably, but they may refer to different things in the frontend.

reactflow [has its own definitions][reactflow-concepts] of "node", "edge" and "handle" which are closely related to InvokeAI graph concepts.

- A reactflow node is related to an InvokeAI invocation. It has a "data" property, which holds the InvokeAI-specific invocation data.
- A reactflow edge is roughly equivalent to an InvokeAI edge.
- A reactflow handle is roughly equivalent to an InvokeAI input or output field.

##### Workflow Linear View

Graphs are very capable data structures, but not everyone wants to work with them all the time.

To allow less technical users - or anyone who wants a less visually noisy workspace - to benefit from the power of nodes, InvokeAI has a workflow feature called the Linear View.

A workflow input field can be added to this Linear View, and its input component can be presented similarly to the Linear UI tabs. Internally, we add the field to the workflow's list of exposed fields.

#### OpenAPI Schema

OpenAPI is a schema specification that can represent complex data structures and relationships. The backend is capable of generating an OpenAPI schema for all invocations.

When the UI connects, it requests this schema and parses each invocation into an **invocation template**. Invocation templates have a number of properties, like title, description and type, but the most important ones are their input and output **field templates**.

Invocation and field templates are the "source of truth" for graphs, because they indicate what the backend is able to process.

When a user adds a new node to their workflow, these templates are used to instantiate a node with fields instantiated from the input and output field templates.

##### Field Instances and Templates

Field templates consist of:

- Name: the identifier of the field, its variable name in python
- Type: derived from the field's type annotation in python (e.g. IntegerField, ImageField, MainModelField)
- Constraints: derived from the field's creation args in python (e.g. minimum value for an integer)
- Default value: optionally provided in the field's creation args (e.g. 42 for an integer)

Field instances are created from the templates and have name, type and optionally a value.

The type of the field determines the UI components that are rendered for it.

A field instance's name associates it with its template.

##### Stateful vs Stateless Fields

**Stateful** fields store their value in the frontend graph. Think primitives, model identifiers, images, etc. Fields are only stateful if the frontend allows the user to directly input a value for them.

Many field types, however, are **stateless**. An example is a `UNetField`, which contains some data describing a UNet. Users cannot directly provide this data - it is created and consumed in the backend.

Stateless fields do not store their value in the node, so their field instances do not have values.

"Custom" fields will always be treated as stateless fields.

##### Collection and Polymorphic Fields

Field types have a name and two flags which may identify it as a **collection** or **polymorphic** field.

If a field is annotated in python as a list, its field type is parsed and flagged as a collection type (e.g. `list[int]`).

If it is annotated as a union of a type and list, the type will be flagged as a polymorphic type (e.g. `Union[int, list[int]]`). Fields may not be unions of different types (e.g. `Union[int, list[str]]` and `Union[int, str]` are not allowed).

## Implementation

The majority of data structures in the backend are [pydantic] models. Pydantic provides OpenAPI schemas for all models and we then generate TypeScript types from those.

The OpenAPI schema is parsed at runtime into our invocation templates.

Workflows and all related data are modeled in the frontend using [zod]. Related types are inferred from the zod schemas.

> In python, invocations are pydantic models with fields. These fields become node inputs. The invocation's `invoke()` function returns a pydantic model - its output. Like the invocation itself, the output model has any number of fields, which become node outputs.

### zod Schemas and Types

The zod schemas, inferred types, and type guards are in [types/].

Roughly order from lowest-level to highest:

- `common.ts`: stateful field data, and couple other misc types
- `field.ts`: fields - types, values, instances, templates
- `invocation.ts`: invocations and other node types
- `workflow.ts`: workflows and constituents

We customize the OpenAPI schema to include additional properties on invocation and field schemas. To facilitate parsing this schema into templates, we modify/wrap the types from [openapi-types] in `openapi.ts`.

### OpenAPI Schema Parsing

The entrypoint for OpenAPI schema parsing is [parseSchema.ts].

General logic flow:

- Iterate over all invocation schema objects
  - Extract relevant invocation-level attributes (e.g. title, type, version, etc)
  - Iterate over the invocation's input fields
    - [Parse each field's type](#parsing-field-types)
    - [Build a field input template](#building-field-input-templates) from the type - either a stateful template or "generic" stateless template
  - Iterate over the invocation's output fields
    - Parse the field's type (same as inputs)
    - [Build a field output template](#building-field-output-templates)
  - Assemble the attributes and fields into an invocation template

Most of these involve very straightforward `reduce`s, but the less intuitive steps are detailed below.

#### Parsing Field Types

Field types are represented as structured objects:

```ts
type FieldType = {
  name: string;
  isCollection: boolean;
  isCollectionOrScalar: boolean;
};
```

The parsing logic is in `parseFieldType.ts`.

There are 4 general cases for field type parsing.

##### Primitive Types

When a field is annotated as a primitive values (e.g. `int`, `str`, `float`), the field type parsing is fairly straightforward. The field is represented by a simple OpenAPI **schema object**, which has a `type` property.

We create a field type name from this `type` string (e.g. `string` -> `StringField`).

##### Complex Types

When a field is annotated as a pydantic model (e.g. `ImageField`, `MainModelField`, `ControlField`), it is represented as a **reference object**. Reference objects are pointers to another schema or reference object within the schema.

We need to **dereference** the schema to pull these out. Dereferencing may require recursion. We use the reference object's name directly for the field type name.

> Unfortunately, at this time, we've had limited success using external libraries to deference at runtime, so we do this ourselves.

##### Collection Types

When a field is annotated as a list of a single type, the schema object has an `items` property. They may be a schema object or reference object and must be parsed to determine the item type.

We use the item type for field type name, adding `isCollection: true` to the field type.

##### Collection or Scalar Types

When a field is annotated as a union of a type and list of that type, the schema object has an `anyOf` property, which holds a list of valid types for the union.

After verifying that the union has two members (a type and list of the same type), we use the type for field type name, adding `isCollectionOrScalar: true` to the field type.

##### Optional Fields

In OpenAPI v3.1, when an object is optional, it is put into an `anyOf` along with a primitive schema object with `type: 'null'`.

Handling this adds a fair bit of complexity, as we now must filter out the `'null'` types and work with the remaining types as described above.

If there is a single remaining schema object, we must recursively call to `parseFieldType()` to get parse it.

#### Building Field Input Templates

Now that we have a field type, we can build an input template for the field.

Stateful fields all get a function to build their template, while stateless fields are constructed directly. This is possible because stateless fields have no default value or constraints.

See [buildFieldInputTemplate.ts].

#### Building Field Output Templates

Field outputs are similar to stateless fields - they do not have any value in the frontend. When building their templates, we don't need a special function for each field type.

See [buildFieldOutputTemplate.ts].

### Managing reactflow State

As described above, the workflow editor state is the essentially the reactflow state, plus some extra metadata.

We provide reactflow with an array of nodes and edges via redux, and a number of [event handlers][reactflow-events]. These handlers dispatch redux actions, managing nodes and edges.

The pieces of redux state relevant to workflows are:

- `state.nodes.nodes`: the reactflow nodes state
- `state.nodes.edges`: the reactflow edges state
- `state.nodes.workflow`: the workflow metadata

#### Building Nodes and Edges

A reactflow node has a few important top-level properties:

- `id`: unique identifier
- `type`: a string that maps to a react component to render the node
- `position`: XY coordinates
- `data`: arbitrary data

When the user adds a node, we build **invocation node data**, storing it in `data`. Invocation properties (e.g. type, version, label, etc.) are copied from the invocation template. Inputs and outputs are built from the invocation template's field templates.

See [buildInvocationNode.ts].

Edges are managed by reactflow, but briefly, they consist of:

- `source`: id of the source node
- `sourceHandle`: id of the source node handle (output field)
- `target`: id of the target node
- `targetHandle`: id of the target node handle (input field)

> Edge creation is gated behind validation logic. This validation compares the input and output field types and overall graph state.

#### Building a Workflow

Building a workflow entity is as simple as dropping the nodes, edges and metadata into an object.

Each node and edge is parsed with a zod schema, which serves to strip out any unneeded data.

See [buildWorkflow.ts].

#### Loading a Workflow

Workflows may be loaded from external sources or the user's local instance. In all cases, the workflow needs to be handled with care, as an untrusted object.

Loading has a few stages which may throw or warn if there are problems:

- Parsing the workflow data structure itself, [migrating](#workflow-migrations) it if necessary (throws)
- Check for a template for each node (warns)
- Check each node's version against its template (warns)
- Validate the source and target of each edge (warns)

This validation occurs in [validateWorkflow.ts].

If there are no fatal errors, the workflow is then stored in redux state.

### Workflow Migrations

When the workflow schema changes, we may need to perform some data migrations. This occurs as workflows are loaded. zod schemas for each workflow schema version is retained to facilitate migrations.

Previous schemas are in folders in `invokeai/frontend/web/src/features/nodes/types/`, eg `v1/`.

Migration logic is in [migrations.ts].

<!-- links -->

[pydantic]: https://github.com/pydantic/pydantic 'pydantic'
[zod]: https://github.com/colinhacks/zod 'zod'
[openapi-types]: https://github.com/kogosoftwarellc/open-api/tree/main/packages/openapi-types 'openapi-types'
[reactflow]: https://github.com/xyflow/xyflow 'reactflow'
[reactflow-concepts]: https://reactflow.dev/learn/concepts/terms-and-definitions
[reactflow-events]: https://reactflow.dev/api-reference/react-flow#event-handlers
[buildWorkflow.ts]: ../src/features/nodes/util/workflow/buildWorkflow.ts
[nodesSlice.ts]: ../src/features/nodes/store/nodesSlice.ts
[buildLinearTextToImageGraph.ts]: ../src/features/nodes/util/graph/buildLinearTextToImageGraph.ts
[buildNodesGraph.ts]: ../src/features/nodes/util/graph/buildNodesGraph.ts
[buildInvocationNode.ts]: ../src/features/nodes/util/node/buildInvocationNode.ts
[validateWorkflow.ts]: ../src/features/nodes/util/workflow/validateWorkflow.ts
[migrations.ts]: ../src/features/nodes/util/workflow/migrations.ts
[parseSchema.ts]: ../src/features/nodes/util/schema/parseSchema.ts
[buildFieldInputTemplate.ts]: ../src/features/nodes/util/schema/buildFieldInputTemplate.ts
[buildFieldOutputTemplate.ts]: ../src/features/nodes/util/schema/buildFieldOutputTemplate.ts
