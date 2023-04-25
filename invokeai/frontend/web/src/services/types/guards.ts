import {
  GraphExecutionState,
  GraphInvocationOutput,
  ImageOutput,
  MaskOutput,
  PromptOutput,
  IterateInvocationOutput,
  CollectInvocationOutput,
} from 'services/api';

export const isImageOutput = (
  output: GraphExecutionState['results'][string]
): output is ImageOutput => output.type === 'image';

export const isMaskOutput = (
  output: GraphExecutionState['results'][string]
): output is MaskOutput => output.type === 'mask';

export const isPromptOutput = (
  output: GraphExecutionState['results'][string]
): output is PromptOutput => output.type === 'prompt';

export const isGraphOutput = (
  output: GraphExecutionState['results'][string]
): output is GraphInvocationOutput => output.type === 'graph_output';

export const isIterateOutput = (
  output: GraphExecutionState['results'][string]
): output is IterateInvocationOutput => output.type === 'iterate_output';

export const isCollectOutput = (
  output: GraphExecutionState['results'][string]
): output is CollectInvocationOutput => output.type === 'collect_output';
