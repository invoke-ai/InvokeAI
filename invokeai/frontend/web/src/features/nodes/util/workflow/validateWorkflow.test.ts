import { img_resize, main_model_loader } from 'features/nodes/store/util/testUtils';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { validateWorkflow } from 'features/nodes/util/workflow/validateWorkflow';
import { get } from 'lodash-es';
import { describe, expect, it } from 'vitest';

describe('validateWorkflow', () => {
  const workflow: WorkflowV3 = {
    name: '',
    author: '',
    description: '',
    version: '',
    contact: '',
    tags: '',
    notes: '',
    exposedFields: [],
    meta: { version: '3.0.0', category: 'user' },
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
              value: { board_id: '99a08f09-8232-4b74-94a2-f8e136d62f8c' },
            },
            metadata: { name: 'metadata', label: 'Metadata' },
            image: {
              name: 'image',
              label: '',
              value: { image_name: '96c124c8-f62f-4d4f-9788-72218469f298.png' },
            },
            width: { name: 'width', label: '', value: 512 },
            height: { name: 'height', label: '', value: 512 },
            resample_mode: { name: 'resample_mode', label: '', value: 'bicubic' },
          },
        },
        position: { x: -46.428806920557236, y: -479.6641524207518 },
      },
    ],
    edges: [],
  };
  const resolveTrue = (): Promise<boolean> =>
    new Promise((resolve) => {
      resolve(true);
    });
  const resolveFalse = (): Promise<boolean> =>
    new Promise((resolve) => {
      resolve(false);
    });
  it('should reset images that are inaccessible', async () => {
    const validationResult = await validateWorkflow(
      workflow,
      { img_resize, main_model_loader },
      resolveFalse,
      resolveTrue,
      resolveTrue
    );
    expect(validationResult.warnings.length).toBe(1);
    expect(get(validationResult, 'workflow.nodes[1].data.inputs.image.value')).toBeUndefined();
  });
  it('should reset boards that are inaccessible', async () => {
    const validationResult = await validateWorkflow(
      workflow,
      { img_resize, main_model_loader },
      resolveTrue,
      resolveFalse,
      resolveTrue
    );
    expect(validationResult.warnings.length).toBe(1);
    expect(get(validationResult, 'workflow.nodes[1].data.inputs.board.value')).toBeUndefined();
  });
  it('should reset models that are inaccessible', async () => {
    const validationResult = await validateWorkflow(
      workflow,
      { img_resize, main_model_loader },
      resolveTrue,
      resolveTrue,
      resolveFalse
    );
    expect(validationResult.warnings.length).toBe(1);
    expect(get(validationResult, 'workflow.nodes[0].data.inputs.model.value')).toBeUndefined();
  });
});
