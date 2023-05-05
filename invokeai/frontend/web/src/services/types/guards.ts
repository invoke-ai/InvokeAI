import { Image } from 'app/types/invokeai';
import { get, isObject, isString } from 'lodash-es';
import {
  GraphExecutionState,
  GraphInvocationOutput,
  ImageOutput,
  MaskOutput,
  PromptOutput,
  IterateInvocationOutput,
  CollectInvocationOutput,
  ImageType,
  ImageField,
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

export const isImageType = (t: unknown): t is ImageType =>
  isString(t) && ['results', 'uploads', 'intermediates'].includes(t);

export const isImage = (image: unknown): image is Image =>
  isObject(image) &&
  isString(get(image, 'name')) &&
  isImageType(get(image, 'type'));

export const isImageField = (imageField: unknown): imageField is ImageField =>
  isObject(imageField) &&
  isString(get(imageField, 'image_name')) &&
  isImageType(get(imageField, 'image_type'));
