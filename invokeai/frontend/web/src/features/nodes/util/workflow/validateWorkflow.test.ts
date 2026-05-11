import { get } from 'es-toolkit/compat';
import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from 'features/nodes/store/util/connectorTopology';
import { buildNode, img_resize, main_model_loader } from 'features/nodes/store/util/testUtils';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { getDefaultForm, isWorkflowInvocationNode } from 'features/nodes/types/workflow';
import { validateWorkflow } from 'features/nodes/util/workflow/validateWorkflow';
import { describe, expect, it } from 'vitest';

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
  const noise: InvocationTemplate = {
    title: 'Create Latent Noise',
    type: 'noise',
    version: '1.1.0',
    tags: ['latents', 'noise'],
    description: 'Generates latent noise for supported denoiser architectures.',
    outputType: 'noise_output',
    inputs: {
      noise_type: {
        name: 'noise_type',
        title: 'Noise Type',
        required: false,
        description: 'Architecture-specific noise type.',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        type: { name: 'EnumField', cardinality: 'SINGLE', batch: false },
        default: 'SD',
        options: ['SD', 'FLUX', 'FLUX.2', 'SD3', 'CogView4', 'Z-Image', 'Anima'],
      },
      seed: {
        name: 'seed',
        title: 'Seed',
        required: false,
        description: 'Seed',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
        default: 0,
      },
      width: {
        name: 'width',
        title: 'Width',
        required: false,
        description: 'Width',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
        default: 512,
      },
      height: {
        name: 'height',
        title: 'Height',
        required: false,
        description: 'Height',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
        default: 512,
      },
      use_cpu: {
        name: 'use_cpu',
        title: 'Use CPU',
        required: false,
        description: 'Use CPU for noise generation',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        type: { name: 'BooleanField', cardinality: 'SINGLE', batch: false },
        default: true,
      },
    },
    outputs: {
      noise: {
        fieldKind: 'output',
        name: 'noise',
        title: 'Noise',
        description: 'Noise output',
        type: { name: 'LatentsField', cardinality: 'SINGLE', batch: false },
        ui_hidden: false,
      },
      width: {
        fieldKind: 'output',
        name: 'width',
        title: 'Width',
        description: 'Width output',
        type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
        ui_hidden: false,
      },
      height: {
        fieldKind: 'output',
        name: 'height',
        title: 'Height',
        description: 'Height output',
        type: { name: 'IntegerField', cardinality: 'SINGLE', batch: false },
        ui_hidden: false,
      },
    },
    useCache: true,
    nodePack: 'invokeai',
    classification: 'stable',
    category: 'latents',
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

  it('should migrate universal_noise nodes to noise and drop the removed transformer input', async () => {
    const workflow = getWorkflow();
    const noiseNode = buildNode(noise);
    const noiseTypeInput = noiseNode.data.inputs.noise_type;
    if (!noiseTypeInput) {
      throw new Error('Missing noise_type input');
    }
    noiseNode.data.type = 'universal_noise';
    noiseNode.data.version = '1.0.0';
    noiseTypeInput.value = 'FLUX';
    noiseNode.data.inputs.transformer = {
      name: 'transformer',
      label: '',
      description: '',
      value: { key: 'transformer-key', hash: 'hash', name: 'name', base: 'sd-3', type: 'main' },
    } as never;
    workflow.nodes = [noiseNode];

    const validationResult = await validateWorkflow({
      workflow,
      templates: { noise },
      checkImageAccess: resolveTrue,
      checkBoardAccess: resolveTrue,
      checkModelAccess: resolveTrue,
    });

    expect(validationResult.warnings).toEqual([]);
    const migratedNode = validationResult.workflow.nodes[0];
    expect(isWorkflowInvocationNode(migratedNode)).toBe(true);
    if (!isWorkflowInvocationNode(migratedNode)) {
      throw new Error('Expected invocation node');
    }
    expect(migratedNode.data.type).toBe('noise');
    expect(migratedNode.data.version).toBe('1.1.0');
    expect(get(validationResult.workflow, 'nodes[0].data.inputs.noise_type.value')).toBe('FLUX');
    expect(get(validationResult.workflow, 'nodes[0].data.inputs.transformer')).toBeUndefined();
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
});
