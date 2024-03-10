import type { EntityState } from '@reduxjs/toolkit';
import type { components, paths } from 'services/api/schema';
import type { O } from 'ts-toolbelt';

export type S = components['schemas'];

export type ImageCache = EntityState<ImageDTO, string>;

export type ListImagesArgs = NonNullable<paths['/api/v1/images/']['get']['parameters']['query']>;

export type DeleteBoardResult =
  paths['/api/v1/boards/{board_id}']['delete']['responses']['200']['content']['application/json'];

export type UpdateBoardArg = paths['/api/v1/boards/{board_id}']['patch']['parameters']['path'] & {
  changes: paths['/api/v1/boards/{board_id}']['patch']['requestBody']['content']['application/json'];
};

export type GraphAndWorkflowResponse =
  paths['/api/v1/images/i/{image_name}/workflow']['get']['responses']['200']['content']['application/json'];

export type BatchConfig =
  paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['requestBody']['content']['application/json'];

export type InputFieldJSONSchemaExtra = S['InputFieldJSONSchemaExtra'];
export type OutputFieldJSONSchemaExtra = S['OutputFieldJSONSchemaExtra'];
export type InvocationJSONSchemaExtra = S['UIConfigBase'];

// App Info
export type AppVersion = S['AppVersion'];
export type AppConfig = S['AppConfig'];
export type AppDependencyVersions = S['AppDependencyVersions'];

// Images
export type ImageDTO = S['ImageDTO'];
export type BoardDTO = S['BoardDTO'];
export type ImageCategory = S['ImageCategory'];
export type OffsetPaginatedResults_ImageDTO_ = S['OffsetPaginatedResults_ImageDTO_'];

// Models
export type ModelType = S['ModelType'];
export type BaseModelType = S['BaseModelType'];

// Model Configs

// TODO(MM2): Can we make key required in the pydantic model?
export type LoRAModelConfig = S['LoRADiffusersConfig'] | S['LoRALyCORISConfig'];
// TODO(MM2): Can we rename this from Vae -> VAE
export type VAEModelConfig = S['VAECheckpointConfig'] | S['VAEDiffusersConfig'];
export type ControlNetModelConfig = S['ControlNetDiffusersConfig'] | S['ControlNetCheckpointConfig'];
export type IPAdapterModelConfig = S['IPAdapterInvokeAIConfig'] | S['IPAdapterCheckpointConfig'];
export type T2IAdapterModelConfig = S['T2IAdapterConfig'];
type TextualInversionModelConfig = S['TextualInversionFileConfig'] | S['TextualInversionFolderConfig'];
type DiffusersModelConfig = S['MainDiffusersConfig'];
type CheckpointModelConfig = S['MainCheckpointConfig'];
type CLIPVisionDiffusersConfig = S['CLIPVisionDiffusersConfig'];
export type MainModelConfig = DiffusersModelConfig | CheckpointModelConfig;
export type AnyModelConfig =
  | LoRAModelConfig
  | VAEModelConfig
  | ControlNetModelConfig
  | IPAdapterModelConfig
  | T2IAdapterModelConfig
  | TextualInversionModelConfig
  | MainModelConfig
  | CLIPVisionDiffusersConfig;

export const isLoRAModelConfig = (config: AnyModelConfig): config is LoRAModelConfig => {
  return config.type === 'lora';
};

export const isVAEModelConfig = (config: AnyModelConfig): config is VAEModelConfig => {
  return config.type === 'vae';
};

export const isControlNetModelConfig = (config: AnyModelConfig): config is ControlNetModelConfig => {
  return config.type === 'controlnet';
};

export const isIPAdapterModelConfig = (config: AnyModelConfig): config is IPAdapterModelConfig => {
  return config.type === 'ip_adapter';
};

export const isT2IAdapterModelConfig = (config: AnyModelConfig): config is T2IAdapterModelConfig => {
  return config.type === 't2i_adapter';
};

export const isControlAdapterModelConfig = (
  config: AnyModelConfig
): config is ControlNetModelConfig | T2IAdapterModelConfig | IPAdapterModelConfig => {
  return isControlNetModelConfig(config) || isT2IAdapterModelConfig(config) || isIPAdapterModelConfig(config);
};

export const isControlNetOrT2IAdapterModelConfig = (
  config: AnyModelConfig
): config is ControlNetModelConfig | T2IAdapterModelConfig => {
  return isControlNetModelConfig(config) || isT2IAdapterModelConfig(config);
};

export const isNonRefinerMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base !== 'sdxl-refiner';
};

export const isRefinerMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl-refiner';
};

export const isSDXLMainModelModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && config.base === 'sdxl';
};

export const isNonSDXLMainModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'main' && (config.base === 'sd-1' || config.base === 'sd-2');
};

export const isTIModelConfig = (config: AnyModelConfig): config is MainModelConfig => {
  return config.type === 'embedding';
};

export type ModelInstallJob = S['ModelInstallJob'];
export type ModelInstallStatus = S['InstallStatus'];

// Graphs
export type Graph = S['Graph'];
export type NonNullableGraph = O.Required<Graph, 'nodes' | 'edges'>;
export type GraphExecutionState = S['GraphExecutionState'];
export type Batch = S['Batch'];
export type SessionQueueItemDTO = S['SessionQueueItemDTO'];
export type WorkflowRecordOrderBy = S['WorkflowRecordOrderBy'];
export type SQLiteDirection = S['SQLiteDirection'];
export type WorkflowRecordListItemDTO = S['WorkflowRecordListItemDTO'];

type KeysOfUnion<T> = T extends T ? keyof T : never;

export type AnyInvocation = Exclude<
  Graph['nodes'][string],
  S['CoreMetadataInvocation'] | S['MetadataInvocation'] | S['MetadataItemInvocation'] | S['MergeMetadataInvocation']
>;
export type AnyInvocationIncMetadata = S['Graph']['nodes'][string];

export type InvocationType = AnyInvocation['type'];
type InvocationOutputMap = S['InvocationOutputMap'];
type AnyInvocationOutput = InvocationOutputMap[InvocationType];

export type Invocation<T extends InvocationType> = Extract<AnyInvocation, { type: T }>;
// export type InvocationOutput<T extends InvocationType> = InvocationOutputMap[T];

type NonInputFields = 'id' | 'type' | 'is_intermediate' | 'use_cache' | 'board' | 'metadata';
export type AnyInvocationInputField = Exclude<KeysOfUnion<Required<AnyInvocation>>, NonInputFields>;
export type InputFields<T extends AnyInvocation> = Extract<keyof T, AnyInvocationInputField>;

type NonOutputFields = 'type';
export type AnyInvocationOutputField = Exclude<KeysOfUnion<Required<AnyInvocationOutput>>, NonOutputFields>;
export type OutputFields<T extends AnyInvocation> = Extract<
  keyof InvocationOutputMap[T['type']],
  AnyInvocationOutputField
>;

// Node Outputs
export type ImageOutput = S['ImageOutput'];

// Post-image upload actions, controls workflows when images are uploaded

type ControlAdapterAction = {
  type: 'SET_CONTROL_ADAPTER_IMAGE';
  id: string;
};

export type CALayerImagePostUploadAction = {
  type: 'SET_CA_LAYER_IMAGE';
  layerId: string;
};

export type IPALayerImagePostUploadAction = {
  type: 'SET_IPA_LAYER_IMAGE';
  layerId: string;
};

export type RGLayerIPAdapterImagePostUploadAction = {
  type: 'SET_RG_LAYER_IP_ADAPTER_IMAGE';
  layerId: string;
  ipAdapterId: string;
};

export type IILayerImagePostUploadAction = {
  type: 'SET_II_LAYER_IMAGE';
  layerId: string;
};

type NodesAction = {
  type: 'SET_NODES_IMAGE';
  nodeId: string;
  fieldName: string;
};

type CanvasInitialImageAction = {
  type: 'SET_CANVAS_INITIAL_IMAGE';
};

type ToastAction = {
  type: 'TOAST';
  title?: string;
};

type AddToBatchAction = {
  type: 'ADD_TO_BATCH';
};

export type PostUploadAction =
  | ControlAdapterAction
  | NodesAction
  | CanvasInitialImageAction
  | ToastAction
  | AddToBatchAction
  | CALayerImagePostUploadAction
  | IPALayerImagePostUploadAction
  | RGLayerIPAdapterImagePostUploadAction
  | IILayerImagePostUploadAction;
