# Workflows - Design and Implementation

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Workflows - Design and Implementation](#workflows---design-and-implementation)
  - [Linear UI](#linear-ui)
  - [Workflow Editor](#workflow-editor)
    - [Workflows](#workflows)
      - [Workflow -> reactflow state -> InvokeAI graph](#workflow---reactflow-state---invokeai-graph)
      - [Nodes vs Invocations](#nodes-vs-invocations)
      - [Workflow Linear View](#workflow-linear-view)
    - [OpenAPI Schema Parsing](#openapi-schema-parsing)
      - [Field Instances and Templates](#field-instances-and-templates)
      - [Stateful vs Stateless Fields](#stateful-vs-stateless-fields)
      - [Collection and Polymorphic Fields](#collection-and-polymorphic-fields)
  - [Implementation](#implementation)

<!-- /code_chunk_output -->

InvokeAI's backend uses graphs, composed of **nodes** and **edges**, to process data and generate images.

Nodes have any number of **input fields** and one **output field**. Edges connect nodes together via their inputs and outputs.

During execution, a nodes' output may be passed along to any number of other nodes' inputs.

We provide two ways to build graphs in the frontend: the [Linear UI](#linear-ui) and [Workflow Editor](#workflow-editor).

## Linear UI

This includes the **Text to Image**, **Image to Image** and **Unified Canvas** tabs.

The user-managed parameters on these tabs are stored as simple objects in the application state. When the user invokes, adding a generation to the queue, we internally build a graph from these parameters.

This logic can be fairly complex due to the range of features available and their interactions. Depending on the parameters selected, the graph may be very different. Building graphs in code can be challenging - you are trying to construct a non-linear structure in a linear context.

The simplest graph building logic is for **Text to Image** with a SD1.5 model:
`invokeai/frontend/web/src/features/nodes/util/graphBuilders/buildLinearTextToImageGraph.ts`

There are many other graph builders in the same folder for different tabs or base models (e.g. SDXL). Some are pretty hairy.

In the Linear UI, we go straight from **simple application state** to **graph** via these builders.

## Workflow Editor

The Workflow Editor is a visual graph editor, allowing users to draw edges from node to node to construct a graph. This _far_ more approachable way to create complex graphs.

InvokeAI uses the [reactflow](https://github.com/xyflow/xyflow) library to power the Workflow Editor. It provides both a graph editor UI and manages its own internal graph state.

### Workflows

So far, we've described two different graph representations used by InvokeAI - the InvokeAI execution graph and the reactflow state.

Neither of these is sufficient to represent a _workflow_, though. A workflow must have a representation of a its graph's nodes and edges, but it also has other data:

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

To support these qualities, workflows are serializable, have a versioned schemas, and represent graphs as minimally as possible. Fortunately, the reactflow state for nodes and edges works perfectly for this..

#### Workflow -> reactflow state -> InvokeAI graph

Given a workflow, we need to be able to derive reactflow state and/or an InvokeAI graph from it.

The first step - workflow to reactflow state - is very simple. The logic is in `invokeai/frontend/web/src/features/nodes/store/nodesSlice.ts`, in the `workflowLoaded` reducer.

The reactflow state is, however, structurally incompatible with our backend's graph structure. When a user invokes on a Workflow, we need to convert the reactflow state into an InvokeAI graph. This is far simpler than the graph building logic from the Linear UI:
`invokeai/frontend/web/src/features/nodes/util/graphBuilders/buildNodesGraph.ts`

#### Nodes vs Invocations

We often use the terms "node" and "invocation" interchangeably, but they may refer to different things in the frontend.

reactflow [has its own definitions](https://reactflow.dev/learn/concepts/terms-and-definitions) of "node", "edge" and "handle" which are closely related to InvokeAI graph concepts.

- A reactflow node is related to an InvokeAI invocation. It has a "data" property, which holds the InvokeAI-specific invocation data.
- A reactflow edge is roughly equivalent to an InvokeAI edge.
- A reactflow handle is roughly equivalent to an InvokeAI input or output field.

#### Workflow Linear View

Graphs are very capable data structures, but not everyone wants to work with them all the time.

To allow less technical users - or anyone who wants a less visually noisy workspace - to benefit from the power of nodes, InvokeAI has a workflow feature called the Linear View.

A workflow input field can be added to this Linear View, and its input component can be presented similarly to the Linear UI tabs. Internally, we add the field to the workflow's list of exposed fields.

### OpenAPI Schema Parsing

OpenAPI is a schema specification that can represent complex data structures and relationships. The backend is capable of generating an OpenAPI schema for all invocations.

When the UI connects, it requests this schema and parses each invocation into an **invocation template**. Invocation templates have a number of properties, like title, description and type, but the most important ones are their input and output **field templates**.

Invocation and field templates are the "source of truth" for graphs, because they indicate what the backend is able to process.

When a user adds a new node to their workflow, these templates are used to instantiate a node with fields instantiated from the input and output field templates.

#### Field Instances and Templates

Field templates consist of:

- Name: the identifier of the field, its variable name in python
- Type: derived from the field's type annotation in python (e.g. IntegerField, ImageField, MainModelField)
- Constraints: derived from the field's creation args in python (e.g. minimum value for an integer)
- Default value: optionally provided in the field's creation args (e.g. 42 for an integer)

Field instances are created from the templates and have name, type and optionally a value.

The type of the field determines the UI components that are rendered for it.

A field instance's name associates it with its template.

#### Stateful vs Stateless Fields

**Stateful** fields store their value in the frontend graph. Think primitives, model identifiers, images, etc. Fields are only stateful if the frontend allows the user to directly input a value for them.

Many field types, however, are **stateless**. An example is a `UNetField`, which contains some data describing a UNet. Users cannot directly provide this data - it is created and consumed in the backend.

Stateless fields do not store their value in the node, so their field instances do not have values.

"Custom" fields will always be treated as stateless fields.

#### Collection and Polymorphic Fields

Field types have a name and two flags which may identify it as a **collection** or **polymorphic** field.

If a field is annotated in python as a list, its field type is parsed and flagged as a collection type (e.g. `list[int]`).

If it is annotated as a union of a type and list, the type will be flagged as a polymorphic type (e.g. `Union[int, list[int]]`). Fields may not be unions of different types (e.g. `Union[int, list[str]]` and `Union[int, str]` are not allowed).

## Implementation

The majority of data structures in the backend are [pydantic](https://github.com/pydantic/pydantic) models. Pydantic provides OpenAPI schemas for all models and we then generate TypeScript types from those.

Workflows and all related data are modeled in the frontend using [zod](https://github.com/colinhacks/zod). Related types are inferred from the zod schemas.

### Schemas and Types

The schemas, inferred types, type guards and related constants are in `invokeai/frontend/web/src/features/nodes/types/`.

Roughly in order from lowest-level to highest:

- `common.ts`: stateful field data, and couple other misc types
- `field.ts`: fields - types, values, instances, templates
- `metadata.ts`: core metadata
- `invocation.ts`: invocations and other node types
- `workflow.ts`: workflows and constituents

### Workflow Migrations
