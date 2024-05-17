import { parseSchema } from 'features/nodes/util/schema/parseSchema';
import { omit, pick } from 'lodash-es';
import type { OpenAPIV3_1 } from 'openapi-types';
import { describe, expect, it } from 'vitest';

describe('parseSchema', () => {
  it('should parse the schema', () => {
    const templates = parseSchema(schema);
    expect(templates).toEqual(expected);
  });
  it('should omit denied nodes', () => {
    const templates = parseSchema(schema, undefined, ['add']);
    expect(templates).toEqual(omit(expected, 'add'));
  });
  it('should include only allowed nodes', () => {
    const templates = parseSchema(schema, ['add']);
    expect(templates).toEqual(pick(expected, 'add'));
  });
});

const expected = {
  add: {
    title: 'Add Integers',
    type: 'add',
    version: '1.0.1',
    tags: ['math', 'add'],
    description: 'Adds two numbers',
    outputType: 'integer_output',
    inputs: {
      a: {
        name: 'a',
        title: 'A',
        required: false,
        description: 'The first number',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        type: {
          name: 'IntegerField',
          isCollection: false,
          isCollectionOrScalar: false,
        },
        default: 0,
      },
      b: {
        name: 'b',
        title: 'B',
        required: false,
        description: 'The second number',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        type: {
          name: 'IntegerField',
          isCollection: false,
          isCollectionOrScalar: false,
        },
        default: 0,
      },
    },
    outputs: {
      value: {
        fieldKind: 'output',
        name: 'value',
        title: 'Value',
        description: 'The output integer',
        type: {
          name: 'IntegerField',
          isCollection: false,
          isCollectionOrScalar: false,
        },
        ui_hidden: false,
      },
    },
    useCache: true,
    nodePack: 'invokeai',
    classification: 'stable',
  },
  scheduler: {
    title: 'Scheduler',
    type: 'scheduler',
    version: '1.0.0',
    tags: ['scheduler'],
    description: 'Selects a scheduler.',
    outputType: 'scheduler_output',
    inputs: {
      scheduler: {
        name: 'scheduler',
        title: 'Scheduler',
        required: false,
        description: 'Scheduler to use during inference',
        fieldKind: 'input',
        input: 'any',
        ui_hidden: false,
        ui_type: 'SchedulerField',
        type: {
          name: 'SchedulerField',
          isCollection: false,
          isCollectionOrScalar: false,
          originalType: {
            name: 'EnumField',
            isCollection: false,
            isCollectionOrScalar: false,
          },
        },
        default: 'euler',
      },
    },
    outputs: {
      scheduler: {
        fieldKind: 'output',
        name: 'scheduler',
        title: 'Scheduler',
        description: 'Scheduler to use during inference',
        type: {
          name: 'SchedulerField',
          isCollection: false,
          isCollectionOrScalar: false,
          originalType: {
            name: 'EnumField',
            isCollection: false,
            isCollectionOrScalar: false,
          },
        },
        ui_hidden: false,
        ui_type: 'SchedulerField',
      },
    },
    useCache: true,
    nodePack: 'invokeai',
    classification: 'stable',
  },
  main_model_loader: {
    title: 'Main Model',
    type: 'main_model_loader',
    version: '1.0.2',
    tags: ['model'],
    description: 'Loads a main model, outputting its submodels.',
    outputType: 'model_loader_output',
    inputs: {
      model: {
        name: 'model',
        title: 'Model',
        required: true,
        description: 'Main model (UNet, VAE, CLIP) to load',
        fieldKind: 'input',
        input: 'direct',
        ui_hidden: false,
        ui_type: 'MainModelField',
        type: {
          name: 'MainModelField',
          isCollection: false,
          isCollectionOrScalar: false,
          originalType: {
            name: 'ModelIdentifierField',
            isCollection: false,
            isCollectionOrScalar: false,
          },
        },
      },
    },
    outputs: {
      vae: {
        fieldKind: 'output',
        name: 'vae',
        title: 'VAE',
        description: 'VAE',
        type: {
          name: 'VAEField',
          isCollection: false,
          isCollectionOrScalar: false,
        },
        ui_hidden: false,
      },
      clip: {
        fieldKind: 'output',
        name: 'clip',
        title: 'CLIP',
        description: 'CLIP (tokenizer, text encoder, LoRAs) and skipped layer count',
        type: {
          name: 'CLIPField',
          isCollection: false,
          isCollectionOrScalar: false,
        },
        ui_hidden: false,
      },
      unet: {
        fieldKind: 'output',
        name: 'unet',
        title: 'UNet',
        description: 'UNet (scheduler, LoRAs)',
        type: {
          name: 'UNetField',
          isCollection: false,
          isCollectionOrScalar: false,
        },
        ui_hidden: false,
      },
    },
    useCache: true,
    nodePack: 'invokeai',
    classification: 'stable',
  },
  collect: {
    title: 'Collect',
    type: 'collect',
    version: '1.0.0',
    tags: [],
    description: 'Collects values into a collection',
    outputType: 'collect_output',
    inputs: {
      item: {
        name: 'item',
        title: 'Collection Item',
        required: false,
        description: 'The item to collect (all inputs must be of the same type)',
        fieldKind: 'input',
        input: 'connection',
        ui_hidden: false,
        ui_type: 'CollectionItemField',
        type: {
          name: 'CollectionItemField',
          isCollection: false,
          isCollectionOrScalar: false,
        },
      },
    },
    outputs: {
      collection: {
        fieldKind: 'output',
        name: 'collection',
        title: 'Collection',
        description: 'The collection of input items',
        type: {
          name: 'CollectionField',
          isCollection: true,
          isCollectionOrScalar: false,
        },
        ui_hidden: false,
        ui_type: 'CollectionField',
      },
    },
    useCache: true,
    classification: 'stable',
  },
};

const schema = {
  openapi: '3.1.0',
  info: {
    title: 'Invoke - Community Edition',
    description: 'An API for invoking AI image operations',
    version: '1.0.0',
  },
  components: {
    schemas: {
      AddInvocation: {
        properties: {
          id: {
            type: 'string',
            title: 'Id',
            description: 'The id of this instance of an invocation. Must be unique among all instances of invocations.',
            field_kind: 'node_attribute',
          },
          is_intermediate: {
            type: 'boolean',
            title: 'Is Intermediate',
            description: 'Whether or not this is an intermediate invocation.',
            default: false,
            field_kind: 'node_attribute',
            ui_type: 'IsIntermediate',
          },
          use_cache: {
            type: 'boolean',
            title: 'Use Cache',
            description: 'Whether or not to use the cache',
            default: true,
            field_kind: 'node_attribute',
          },
          a: {
            type: 'integer',
            title: 'A',
            description: 'The first number',
            default: 0,
            field_kind: 'input',
            input: 'any',
            orig_default: 0,
            orig_required: false,
            ui_hidden: false,
          },
          b: {
            type: 'integer',
            title: 'B',
            description: 'The second number',
            default: 0,
            field_kind: 'input',
            input: 'any',
            orig_default: 0,
            orig_required: false,
            ui_hidden: false,
          },
          type: {
            type: 'string',
            enum: ['add'],
            const: 'add',
            title: 'type',
            default: 'add',
            field_kind: 'node_attribute',
          },
        },
        type: 'object',
        required: ['type', 'id'],
        title: 'Add Integers',
        description: 'Adds two numbers',
        category: 'math',
        classification: 'stable',
        node_pack: 'invokeai',
        tags: ['math', 'add'],
        version: '1.0.1',
        output: {
          $ref: '#/components/schemas/IntegerOutput',
        },
        class: 'invocation',
      },
      IntegerOutput: {
        description: 'Base class for nodes that output a single integer',
        properties: {
          value: {
            description: 'The output integer',
            field_kind: 'output',
            title: 'Value',
            type: 'integer',
            ui_hidden: false,
          },
          type: {
            const: 'integer_output',
            default: 'integer_output',
            enum: ['integer_output'],
            field_kind: 'node_attribute',
            title: 'type',
            type: 'string',
          },
        },
        required: ['value', 'type', 'type'],
        title: 'IntegerOutput',
        type: 'object',
        class: 'output',
      },
      SchedulerInvocation: {
        properties: {
          id: {
            type: 'string',
            title: 'Id',
            description: 'The id of this instance of an invocation. Must be unique among all instances of invocations.',
            field_kind: 'node_attribute',
          },
          is_intermediate: {
            type: 'boolean',
            title: 'Is Intermediate',
            description: 'Whether or not this is an intermediate invocation.',
            default: false,
            field_kind: 'node_attribute',
            ui_type: 'IsIntermediate',
          },
          use_cache: {
            type: 'boolean',
            title: 'Use Cache',
            description: 'Whether or not to use the cache',
            default: true,
            field_kind: 'node_attribute',
          },
          scheduler: {
            type: 'string',
            enum: [
              'ddim',
              'ddpm',
              'deis',
              'lms',
              'lms_k',
              'pndm',
              'heun',
              'heun_k',
              'euler',
              'euler_k',
              'euler_a',
              'kdpm_2',
              'kdpm_2_a',
              'dpmpp_2s',
              'dpmpp_2s_k',
              'dpmpp_2m',
              'dpmpp_2m_k',
              'dpmpp_2m_sde',
              'dpmpp_2m_sde_k',
              'dpmpp_sde',
              'dpmpp_sde_k',
              'unipc',
              'lcm',
              'tcd',
            ],
            title: 'Scheduler',
            description: 'Scheduler to use during inference',
            default: 'euler',
            field_kind: 'input',
            input: 'any',
            orig_default: 'euler',
            orig_required: false,
            ui_hidden: false,
            ui_type: 'SchedulerField',
          },
          type: {
            type: 'string',
            enum: ['scheduler'],
            const: 'scheduler',
            title: 'type',
            default: 'scheduler',
            field_kind: 'node_attribute',
          },
        },
        type: 'object',
        required: ['type', 'id'],
        title: 'Scheduler',
        description: 'Selects a scheduler.',
        category: 'latents',
        classification: 'stable',
        node_pack: 'invokeai',
        tags: ['scheduler'],
        version: '1.0.0',
        output: {
          $ref: '#/components/schemas/SchedulerOutput',
        },
        class: 'invocation',
      },
      SchedulerOutput: {
        properties: {
          scheduler: {
            description: 'Scheduler to use during inference',
            enum: [
              'ddim',
              'ddpm',
              'deis',
              'lms',
              'lms_k',
              'pndm',
              'heun',
              'heun_k',
              'euler',
              'euler_k',
              'euler_a',
              'kdpm_2',
              'kdpm_2_a',
              'dpmpp_2s',
              'dpmpp_2s_k',
              'dpmpp_2m',
              'dpmpp_2m_k',
              'dpmpp_2m_sde',
              'dpmpp_2m_sde_k',
              'dpmpp_sde',
              'dpmpp_sde_k',
              'unipc',
              'lcm',
              'tcd',
            ],
            field_kind: 'output',
            title: 'Scheduler',
            type: 'string',
            ui_hidden: false,
            ui_type: 'SchedulerField',
          },
          type: {
            const: 'scheduler_output',
            default: 'scheduler_output',
            enum: ['scheduler_output'],
            field_kind: 'node_attribute',
            title: 'type',
            type: 'string',
          },
        },
        required: ['scheduler', 'type', 'type'],
        title: 'SchedulerOutput',
        type: 'object',
        class: 'output',
      },
      MainModelLoaderInvocation: {
        properties: {
          id: {
            type: 'string',
            title: 'Id',
            description: 'The id of this instance of an invocation. Must be unique among all instances of invocations.',
            field_kind: 'node_attribute',
          },
          is_intermediate: {
            type: 'boolean',
            title: 'Is Intermediate',
            description: 'Whether or not this is an intermediate invocation.',
            default: false,
            field_kind: 'node_attribute',
            ui_type: 'IsIntermediate',
          },
          use_cache: {
            type: 'boolean',
            title: 'Use Cache',
            description: 'Whether or not to use the cache',
            default: true,
            field_kind: 'node_attribute',
          },
          model: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelIdentifierField',
              },
            ],
            description: 'Main model (UNet, VAE, CLIP) to load',
            field_kind: 'input',
            input: 'direct',
            orig_required: true,
            ui_hidden: false,
            ui_type: 'MainModelField',
          },
          type: {
            type: 'string',
            enum: ['main_model_loader'],
            const: 'main_model_loader',
            title: 'type',
            default: 'main_model_loader',
            field_kind: 'node_attribute',
          },
        },
        type: 'object',
        required: ['model', 'type', 'id'],
        title: 'Main Model',
        description: 'Loads a main model, outputting its submodels.',
        category: 'model',
        classification: 'stable',
        node_pack: 'invokeai',
        tags: ['model'],
        version: '1.0.2',
        output: {
          $ref: '#/components/schemas/ModelLoaderOutput',
        },
        class: 'invocation',
      },
      ModelIdentifierField: {
        properties: {
          key: {
            description: "The model's unique key",
            title: 'Key',
            type: 'string',
          },
          hash: {
            description: "The model's BLAKE3 hash",
            title: 'Hash',
            type: 'string',
          },
          name: {
            description: "The model's name",
            title: 'Name',
            type: 'string',
          },
          base: {
            allOf: [
              {
                $ref: '#/components/schemas/BaseModelType',
              },
            ],
            description: "The model's base model type",
          },
          type: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelType',
              },
            ],
            description: "The model's type",
          },
          submodel_type: {
            anyOf: [
              {
                $ref: '#/components/schemas/SubModelType',
              },
              {
                type: 'null',
              },
            ],
            default: null,
            description: 'The submodel to load, if this is a main model',
          },
        },
        required: ['key', 'hash', 'name', 'base', 'type'],
        title: 'ModelIdentifierField',
        type: 'object',
      },
      BaseModelType: {
        description: 'Base model type.',
        enum: ['any', 'sd-1', 'sd-2', 'sdxl', 'sdxl-refiner'],
        title: 'BaseModelType',
        type: 'string',
      },
      ModelType: {
        description: 'Model type.',
        enum: ['onnx', 'main', 'vae', 'lora', 'controlnet', 'embedding', 'ip_adapter', 'clip_vision', 't2i_adapter'],
        title: 'ModelType',
        type: 'string',
      },
      SubModelType: {
        description: 'Submodel type.',
        enum: [
          'unet',
          'text_encoder',
          'text_encoder_2',
          'tokenizer',
          'tokenizer_2',
          'vae',
          'vae_decoder',
          'vae_encoder',
          'scheduler',
          'safety_checker',
        ],
        title: 'SubModelType',
        type: 'string',
      },
      ModelLoaderOutput: {
        description: 'Model loader output',
        properties: {
          vae: {
            allOf: [
              {
                $ref: '#/components/schemas/VAEField',
              },
            ],
            description: 'VAE',
            field_kind: 'output',
            title: 'VAE',
            ui_hidden: false,
          },
          type: {
            const: 'model_loader_output',
            default: 'model_loader_output',
            enum: ['model_loader_output'],
            field_kind: 'node_attribute',
            title: 'type',
            type: 'string',
          },
          clip: {
            allOf: [
              {
                $ref: '#/components/schemas/CLIPField',
              },
            ],
            description: 'CLIP (tokenizer, text encoder, LoRAs) and skipped layer count',
            field_kind: 'output',
            title: 'CLIP',
            ui_hidden: false,
          },
          unet: {
            allOf: [
              {
                $ref: '#/components/schemas/UNetField',
              },
            ],
            description: 'UNet (scheduler, LoRAs)',
            field_kind: 'output',
            title: 'UNet',
            ui_hidden: false,
          },
        },
        required: ['vae', 'type', 'clip', 'unet', 'type'],
        title: 'ModelLoaderOutput',
        type: 'object',
        class: 'output',
      },
      UNetField: {
        properties: {
          unet: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelIdentifierField',
              },
            ],
            description: 'Info to load unet submodel',
          },
          scheduler: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelIdentifierField',
              },
            ],
            description: 'Info to load scheduler submodel',
          },
          loras: {
            description: 'LoRAs to apply on model loading',
            items: {
              $ref: '#/components/schemas/LoRAField',
            },
            title: 'Loras',
            type: 'array',
          },
          seamless_axes: {
            description: 'Axes("x" and "y") to which apply seamless',
            items: {
              type: 'string',
            },
            title: 'Seamless Axes',
            type: 'array',
          },
          freeu_config: {
            anyOf: [
              {
                $ref: '#/components/schemas/FreeUConfig',
              },
              {
                type: 'null',
              },
            ],
            default: null,
            description: 'FreeU configuration',
          },
        },
        required: ['unet', 'scheduler', 'loras'],
        title: 'UNetField',
        type: 'object',
        class: 'output',
      },
      LoRAField: {
        properties: {
          lora: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelIdentifierField',
              },
            ],
            description: 'Info to load lora model',
          },
          weight: {
            description: 'Weight to apply to lora model',
            title: 'Weight',
            type: 'number',
          },
        },
        required: ['lora', 'weight'],
        title: 'LoRAField',
        type: 'object',
        class: 'output',
      },
      FreeUConfig: {
        description:
          'Configuration for the FreeU hyperparameters.\n- https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu\n- https://github.com/ChenyangSi/FreeU',
        properties: {
          s1: {
            description:
              'Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.',
            maximum: 3.0,
            minimum: -1.0,
            title: 'S1',
            type: 'number',
          },
          s2: {
            description:
              'Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.',
            maximum: 3.0,
            minimum: -1.0,
            title: 'S2',
            type: 'number',
          },
          b1: {
            description: 'Scaling factor for stage 1 to amplify the contributions of backbone features.',
            maximum: 3.0,
            minimum: -1.0,
            title: 'B1',
            type: 'number',
          },
          b2: {
            description: 'Scaling factor for stage 2 to amplify the contributions of backbone features.',
            maximum: 3.0,
            minimum: -1.0,
            title: 'B2',
            type: 'number',
          },
        },
        required: ['s1', 's2', 'b1', 'b2'],
        title: 'FreeUConfig',
        type: 'object',
        class: 'output',
      },
      VAEField: {
        properties: {
          vae: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelIdentifierField',
              },
            ],
            description: 'Info to load vae submodel',
          },
          seamless_axes: {
            description: 'Axes("x" and "y") to which apply seamless',
            items: {
              type: 'string',
            },
            title: 'Seamless Axes',
            type: 'array',
          },
        },
        required: ['vae'],
        title: 'VAEField',
        type: 'object',
        class: 'output',
      },
      CLIPField: {
        properties: {
          tokenizer: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelIdentifierField',
              },
            ],
            description: 'Info to load tokenizer submodel',
          },
          text_encoder: {
            allOf: [
              {
                $ref: '#/components/schemas/ModelIdentifierField',
              },
            ],
            description: 'Info to load text_encoder submodel',
          },
          skipped_layers: {
            description: 'Number of skipped layers in text_encoder',
            title: 'Skipped Layers',
            type: 'integer',
          },
          loras: {
            description: 'LoRAs to apply on model loading',
            items: {
              $ref: '#/components/schemas/LoRAField',
            },
            title: 'Loras',
            type: 'array',
          },
        },
        required: ['tokenizer', 'text_encoder', 'skipped_layers', 'loras'],
        title: 'CLIPField',
        type: 'object',
        class: 'output',
      },
      CollectInvocation: {
        properties: {
          id: {
            type: 'string',
            title: 'Id',
            description: 'The id of this instance of an invocation. Must be unique among all instances of invocations.',
            field_kind: 'node_attribute',
          },
          is_intermediate: {
            type: 'boolean',
            title: 'Is Intermediate',
            description: 'Whether or not this is an intermediate invocation.',
            default: false,
            field_kind: 'node_attribute',
            ui_type: 'IsIntermediate',
          },
          use_cache: {
            type: 'boolean',
            title: 'Use Cache',
            description: 'Whether or not to use the cache',
            default: true,
            field_kind: 'node_attribute',
          },
          item: {
            anyOf: [
              {},
              {
                type: 'null',
              },
            ],
            title: 'Collection Item',
            description: 'The item to collect (all inputs must be of the same type)',
            field_kind: 'input',
            input: 'connection',
            orig_required: false,
            ui_hidden: false,
            ui_type: 'CollectionItemField',
          },
          collection: {
            items: {},
            type: 'array',
            title: 'Collection',
            description: 'The collection, will be provided on execution',
            default: [],
            field_kind: 'input',
            input: 'any',
            orig_default: [],
            orig_required: false,
            ui_hidden: true,
          },
          type: {
            type: 'string',
            enum: ['collect'],
            const: 'collect',
            title: 'type',
            default: 'collect',
            field_kind: 'node_attribute',
          },
        },
        type: 'object',
        required: ['type', 'id'],
        title: 'CollectInvocation',
        description: 'Collects values into a collection',
        classification: 'stable',
        version: '1.0.0',
        output: {
          $ref: '#/components/schemas/CollectInvocationOutput',
        },
        class: 'invocation',
      },
      CollectInvocationOutput: {
        properties: {
          collection: {
            description: 'The collection of input items',
            field_kind: 'output',
            items: {},
            title: 'Collection',
            type: 'array',
            ui_hidden: false,
            ui_type: 'CollectionField',
          },
          type: {
            const: 'collect_output',
            default: 'collect_output',
            enum: ['collect_output'],
            field_kind: 'node_attribute',
            title: 'type',
            type: 'string',
          },
        },
        required: ['collection', 'type', 'type'],
        title: 'CollectInvocationOutput',
        type: 'object',
        class: 'output',
      },
    },
  },
} as OpenAPIV3_1.Document;
