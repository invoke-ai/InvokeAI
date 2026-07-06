import { get } from 'es-toolkit/compat';
import { addElement } from 'features/nodes/components/sidePanel/builder/form-manipulation';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from 'features/nodes/store/util/connectorTopology';
import { img_resize, main_model_loader } from 'features/nodes/store/util/testUtils';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { buildNodeFieldElement, getDefaultForm, isNodeFieldElement } from 'features/nodes/types/workflow';
import { validateWorkflow } from 'features/nodes/util/workflow/validateWorkflow';
import { describe, expect, it } from 'vitest';

const imageCollectionTemplate = {
  title: 'Image Collection Primitive',
  type: 'image_collection',
  version: '1.0.2',
  tags: ['primitives', 'image', 'collection'],
  description: 'A collection of image primitive values',
  outputType: 'image_collection_output',
  inputs: {
    collection: {
      name: 'collection',
      title: 'Collection',
      required: false,
      description: 'An optional image collection to append to',
      fieldKind: 'input',
      input: 'connection',
      ui_hidden: false,
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      default: undefined,
    },
    images: {
      name: 'images',
      title: 'Images',
      required: false,
      description: 'The images to append to the collection',
      fieldKind: 'input',
      input: 'direct',
      ui_hidden: false,
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      default: undefined,
    },
  },
  outputs: {
    collection: {
      fieldKind: 'output',
      name: 'collection',
      title: 'Collection',
      description: 'The output images',
      type: { name: 'ImageField', cardinality: 'COLLECTION', batch: false },
      ui_hidden: false,
    },
  },
  useCache: true,
  nodePack: 'invokeai',
  classification: 'stable',
  category: 'primitives',
} satisfies InvocationTemplate;

//TODO(psyche): Test workflow validation for form builder fields
describe('validateWorkflow', () => {
  const buildConnectorNode = (id: string) => ({
    id,
    type: 'connector' as const,
    position: { x: 0, y: 0 },
    data: {
      id,
      type: 'connector' as const,
      label: 'Connector',
      isOpen: true,
    },
  });
  const getWorkflow = (): WorkflowV3 => ({
    name: '',
    author: '',
    description: '',
    version: '',
    contact: '',
    tags: '',
    notes: '',
    exposedFields: [],
    form: getDefaultForm(),
    meta: { version: '4.0.0', category: 'user' },
    nodes: [
      {
        id: '94b1d596-f2f2-4c1c-bd5b-a79c62d947ad',
        type: 'invocation',
        data: {
          id: '94b1d596-f2f2-4c1c-bd5b-a79c62d947ad',
          type: 'main_model_loader',
          version: '1.0.2',
          label: '',
          notes: '',
          isOpen: true,
          isIntermediate: true,
          useCache: true,
          nodePack: 'invokeai',
          inputs: {
            model: {
              name: 'model',
              label: '',
              description: '',
              value: {
                key: '2c85d9e7-12cd-4e59-bb94-96d4502e99d4',
                hash: 'random:aadc6641321ba17a324788ef1691f3584b382f0e7fa4a90be169f2a4ac77435c',
                name: 'Analog-Diffusion2',
                base: 'sd-1',
                type: 'main',
              },
            },
          },
        },
        position: { x: 394.62314170481613, y: -424.6962537790139 },
      },
      {
        id: 'afad11b4-bb5c-45d1-b956-6c8e2357ee11',
        type: 'invocation',
        data: {
          id: 'afad11b4-bb5c-45d1-b956-6c8e2357ee11',
          type: 'img_resize',
          version: '1.2.2',
          label: '',
          notes: '',
          isOpen: true,
          isIntermediate: true,
          useCache: true,
          nodePack: 'invokeai',
          inputs: {
            board: {
              name: 'board',
              label: '',
              description: '',
              value: { board_id: '99a08f09-8232-4b74-94a2-f8e136d62f8c' },
            },
            metadata: { name: 'metadata', label: 'Metadata', description: '' },
            image: {
              name: 'image',
              label: '',
              description: '',
              value: { image_name: '96c124c8-f62f-4d4f-9788-72218469f298.png' },
            },
            width: { name: 'width', label: '', value: 512, description: '' },
            height: { name: 'height', label: '', value: 512, description: '' },
            resample_mode: { name: 'resample_mode', label: '', value: 'bicubic', description: '' },
          },
        },
        position: { x: -46.428806920557236, y: -479.6641524207518 },
      },
    ],
    edges: [],
  });
  const resolveTrue = (): Promise<boolean> =>
    new Promise((resolve) => {
      resolve(true);
    });
  const resolveFalse = (): Promise<boolean> =>
    new Promise((resolve) => {
      resolve(false);
    });
  const addLegacyImageCollectionNode = (workflow: WorkflowV3, id: string, images = [{ image_name: `${id}.png` }]) => {
    workflow.nodes.push({
      id,
      type: 'invocation',
      data: {
        id,
        type: 'image_collection',
        version: '1.0.1',
        label: '',
        notes: '',
        isOpen: true,
        isIntermediate: true,
        useCache: true,
        nodePack: 'invokeai',
        inputs: {
          collection: {
            name: 'collection',
            label: '',
            description: '',
            value: images,
          },
        },
      },
      position: { x: 0, y: 0 },
    });
  };
  it('should reset images that are inaccessible', async () => {
    const validationResult = await validateWorkflow({
      workflow: getWorkflow(),
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveFalse,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });
    expect(validationResult.warnings.length).toBe(1);
    expect(get(validationResult, 'workflow.nodes[1].data.inputs.image.value')).toBeUndefined();
    expect(validationResult.workflow.meta.version).toBe('4.0.0');
  });
  it('should reset boards that are inaccessible', async () => {
    const validationResult = await validateWorkflow({
      workflow: getWorkflow(),
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveFalse,
      checkModelAccess: resolveTrue,
    });
    expect(validationResult.warnings.length).toBe(1);
    expect(get(validationResult, 'workflow.nodes[1].data.inputs.board.value')).toBeUndefined();
  });
  it('should reset models that are inaccessible', async () => {
    const validationResult = await validateWorkflow({
      workflow: getWorkflow(),
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveFalse,
    });
    expect(validationResult.warnings.length).toBe(1);
    expect(get(validationResult, 'workflow.nodes[0].data.inputs.model.value')).toBeUndefined();
  });

  it('should delete malformed connector edges with invalid handles', async () => {
    const workflow = getWorkflow();
    workflow.nodes.push(buildConnectorNode('connector-1'));
    workflow.edges.push({
      id: 'e1',
      type: 'default',
      source: workflow.nodes[0]!.id,
      sourceHandle: 'vae',
      target: 'connector-1',
      targetHandle: 'wrong',
    });

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.edges).toEqual([]);
    expect(validationResult.warnings.length).toBe(1);
  });

  it('should delete connector edges with missing endpoints', async () => {
    const workflow = getWorkflow();
    workflow.nodes.push(buildConnectorNode('connector-1'));
    workflow.edges.push({
      id: 'e1',
      type: 'default',
      source: 'missing-node',
      sourceHandle: 'value',
      target: 'connector-1',
      targetHandle: CONNECTOR_INPUT_HANDLE,
    });

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.edges).toEqual([]);
    expect(validationResult.warnings.length).toBe(1);
  });

  it('should repair invalid multi-input connector state predictably by keeping the first valid input edge', async () => {
    const workflow = getWorkflow();
    const loader2 = structuredClone(workflow.nodes[0]!);
    loader2.id = 'second-loader';
    loader2.data.id = 'second-loader';
    const connector = buildConnectorNode('connector-1');
    workflow.nodes.push(loader2, connector);
    workflow.edges.push({
      id: 'e1',
      type: 'default',
      source: workflow.nodes[0]!.id,
      sourceHandle: 'vae',
      target: connector.id,
      targetHandle: CONNECTOR_INPUT_HANDLE,
    });
    workflow.edges.push({
      id: 'e2',
      type: 'default',
      source: loader2.id,
      sourceHandle: 'vae',
      target: connector.id,
      targetHandle: CONNECTOR_INPUT_HANDLE,
    });

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.edges).toEqual([
      {
        id: 'e1',
        type: 'default',
        source: workflow.nodes[0]!.id,
        sourceHandle: 'vae',
        target: connector.id,
        targetHandle: CONNECTOR_INPUT_HANDLE,
      },
    ]);
    expect(validationResult.warnings.length).toBe(1);
  });

  it('should retain isolated connectors during workflow validation', async () => {
    const workflow = getWorkflow();
    workflow.nodes.push(buildConnectorNode('connector-1'));

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.nodes.find((node) => node.id === 'connector-1')).toBeDefined();
    expect(validationResult.warnings).toEqual([]);
  });

  it('should retain unresolved connector output edges that establish downstream constraints in the editor', async () => {
    const workflow = getWorkflow();
    workflow.nodes.push(buildConnectorNode('connector-1'));
    const unresolvedEdge = {
      id: 'e1',
      type: 'default' as const,
      source: 'connector-1',
      sourceHandle: CONNECTOR_OUTPUT_HANDLE,
      target: workflow.nodes[1]!.id,
      targetHandle: 'image',
    };
    workflow.edges.push(unresolvedEdge);

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.edges).toEqual([unresolvedEdge]);
    expect(validationResult.warnings).toEqual([]);
  });

  it('should migrate image_collection direct values when its old collection edge is invalid', async () => {
    const workflow = getWorkflow();
    const images = [{ image_name: 'legacy.png' }];
    addLegacyImageCollectionNode(workflow, 'image-collection', images);
    workflow.edges.push({
      id: 'missing-source-edge',
      type: 'default',
      source: 'missing-node',
      sourceHandle: 'collection',
      target: 'image-collection',
      targetHandle: 'collection',
    });

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader, image_collection: imageCollectionTemplate },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.edges).toEqual([]);
    expect(get(validationResult.workflow, 'nodes[2].data.inputs.images.value')).toEqual(images);
    expect(get(validationResult.workflow, 'nodes[2].data.inputs.collection.value')).toEqual([]);
  });

  it('should preserve image_collection collection values when its old collection edge is valid', async () => {
    const workflow = getWorkflow();
    const images = [{ image_name: 'shadowed.png' }];
    addLegacyImageCollectionNode(workflow, 'source-collection', []);
    addLegacyImageCollectionNode(workflow, 'target-collection', images);
    workflow.edges.push({
      id: 'valid-edge',
      type: 'default',
      source: 'source-collection',
      sourceHandle: 'collection',
      target: 'target-collection',
      targetHandle: 'collection',
    });

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader, image_collection: imageCollectionTemplate },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.edges).toHaveLength(1);
    expect(get(validationResult.workflow, 'nodes[3].data.inputs.images.value')).toBeUndefined();
    expect(get(validationResult.workflow, 'nodes[3].data.inputs.collection.value')).toEqual(images);
  });

  it('should remap image_collection form and exposed collection fields to images', async () => {
    const workflow = getWorkflow();
    const images = [{ image_name: 'legacy.png' }];
    addLegacyImageCollectionNode(workflow, 'image-collection', images);
    workflow.exposedFields = [{ nodeId: 'image-collection', fieldName: 'collection' }];
    const element = buildNodeFieldElement('image-collection', 'collection', {
      name: 'ImageField',
      cardinality: 'COLLECTION',
      batch: false,
    });
    addElement({ form: workflow.form, element, parentId: workflow.form.rootElementId });

    const validationResult = await validateWorkflow({
      workflow,
      templates: { img_resize, main_model_loader, image_collection: imageCollectionTemplate },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.workflow.exposedFields).toEqual([{ nodeId: 'image-collection', fieldName: 'images' }]);
    const updatedElement = validationResult.workflow.form.elements[element.id];
    if (!updatedElement || !isNodeFieldElement(updatedElement)) {
      throw new Error('Expected a node field form element');
    }
    expect(updatedElement.data.fieldIdentifier.fieldName).toBe('images');
  });
});
