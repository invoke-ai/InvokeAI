import { forEach, size } from 'lodash-es';
import {
  ImageField,
  LatentsField,
  ConditioningField,
  UNetField,
  ClipField,
  VaeField,
} from 'services/api';

const OBJECT_TYPESTRING = '[object Object]';
const STRING_TYPESTRING = '[object String]';
const NUMBER_TYPESTRING = '[object Number]';
const BOOLEAN_TYPESTRING = '[object Boolean]';
const ARRAY_TYPESTRING = '[object Array]';

const isObject = (obj: unknown): obj is Record<string | number, any> =>
  Object.prototype.toString.call(obj) === OBJECT_TYPESTRING;

const isString = (obj: unknown): obj is string =>
  Object.prototype.toString.call(obj) === STRING_TYPESTRING;

const isNumber = (obj: unknown): obj is number =>
  Object.prototype.toString.call(obj) === NUMBER_TYPESTRING;

const isBoolean = (obj: unknown): obj is boolean =>
  Object.prototype.toString.call(obj) === BOOLEAN_TYPESTRING;

const isArray = (obj: unknown): obj is Array<any> =>
  Object.prototype.toString.call(obj) === ARRAY_TYPESTRING;

const parseImageField = (imageField: unknown): ImageField | undefined => {
  // Must be an object
  if (!isObject(imageField)) {
    return;
  }

  // An ImageField must have both `image_name` and `image_type`
  if (!('image_name' in imageField && 'image_type' in imageField)) {
    return;
  }

  // An ImageField's `image_type` must be one of the allowed values
  if (
    !['results', 'uploads', 'intermediates'].includes(imageField.image_type)
  ) {
    return;
  }

  // An ImageField's `image_name` must be a string
  if (typeof imageField.image_name !== 'string') {
    return;
  }

  // Build a valid ImageField
  return {
    image_type: imageField.image_type,
    image_name: imageField.image_name,
  };
};

const parseLatentsField = (latentsField: unknown): LatentsField | undefined => {
  // Must be an object
  if (!isObject(latentsField)) {
    return;
  }

  // A LatentsField must have a `latents_name`
  if (!('latents_name' in latentsField)) {
    return;
  }

  // A LatentsField's `latents_name` must be a string
  if (typeof latentsField.latents_name !== 'string') {
    return;
  }

  // Build a valid LatentsField
  return {
    latents_name: latentsField.latents_name,
  };
};

const parseConditioningField = (
  conditioningField: unknown
): ConditioningField | undefined => {
  // Must be an object
  if (!isObject(conditioningField)) {
    return;
  }

  // A ConditioningField must have a `conditioning_name`
  if (!('conditioning_name' in conditioningField)) {
    return;
  }

  // A ConditioningField's `conditioning_name` must be a string
  if (typeof conditioningField.conditioning_name !== 'string') {
    return;
  }

  // Build a valid ConditioningField
  return {
    conditioning_name: conditioningField.conditioning_name,
  };
};

const _parseModelInfo = (modelInfo: unknown): ModelInfo | undefined => {
  // Must be an object
  if (!isObject(modelInfo)) {
    return;
  }

  if (!('model_name' in modelInfo && typeof modelInfo.model_name == 'string')) {
    return;
  }

  if (!('model_type' in modelInfo && typeof modelInfo.model_type == 'string')) {
    return;
  }

  if (!('submodel' in modelInfo && typeof modelInfo.submodel == 'string')) {
    return;
  }

  return {
    model_name: modelInfo.model_name,
    model_type: modelInfo.model_type,
    submodel: modelInfo.submodel,
  };
};

const parseUNetField = (unetField: unknown): UNetField | undefined => {
  // Must be an object
  if (!isObject(unetField)) {
    return;
  }

  if (!('unet' in unetField && 'scheduler' in unetField)) {
    return;
  }

  const unet = _parseModelInfo(unetField.unet);
  const scheduler = _parseModelInfo(unetField.scheduler);

  if (!(unet && scheduler)) {
    return;
  }

  // Build a valid UNetField
  return {
    unet: unet,
    scheduler: scheduler,
  };
};

const parseClipField = (clipField: unknown): ClipField | undefined => {
  // Must be an object
  if (!isObject(clipField)) {
    return;
  }

  if (!('tokenizer' in clipField && 'text_encoder' in clipField)) {
    return;
  }

  const tokenizer = _parseModelInfo(clipField.tokenizer);
  const text_encoder = _parseModelInfo(clipField.text_encoder);

  if (!(tokenizer && text_encoder)) {
    return;
  }

  // Build a valid ClipField
  return {
    tokenizer: tokenizer,
    text_encoder: text_encoder,
  };
};

const parseVaeField = (vaeField: unknown): VaeField | undefined => {
  // Must be an object
  if (!isObject(vaeField)) {
    return;
  }

  if (!('vae' in vaeField)) {
    return;
  }

  const vae = _parseModelInfo(vaeField.vae);

  if (!vae) {
    return;
  }

  // Build a valid VaeField
  return {
    vae: vae,
  };
};

type NodeMetadata = {
  [key: string]:
    | string
    | number
    | boolean
    | ImageField
    | LatentsField
    | ConditioningField
    | UNetField
    | ClipField
    | VaeField;
};

type InvokeAIMetadata = {
  session_id?: string;
  node?: NodeMetadata;
};

export const parseNodeMetadata = (
  nodeMetadata: Record<string | number, any>
): NodeMetadata | undefined => {
  if (!isObject(nodeMetadata)) {
    return;
  }

  const parsed: NodeMetadata = {};

  forEach(nodeMetadata, (nodeItem, nodeKey) => {
    // `id` and `type` must be strings if they are present
    if (['id', 'type'].includes(nodeKey)) {
      if (isString(nodeItem)) {
        parsed[nodeKey] = nodeItem;
      }
      return;
    }

    // valid object types are:
    // ImageField, LatentsField ConditioningField, UNetField, ClipField, VaeField
    if (isObject(nodeItem)) {
      if ('image_name' in nodeItem || 'image_type' in nodeItem) {
        const imageField = parseImageField(nodeItem);
        if (imageField) {
          parsed[nodeKey] = imageField;
        }
        return;
      }

      if ('latents_name' in nodeItem) {
        const latentsField = parseLatentsField(nodeItem);
        if (latentsField) {
          parsed[nodeKey] = latentsField;
        }
        return;
      }

      if ('conditioning_name' in nodeItem) {
        const conditioningField = parseConditioningField(nodeItem);
        if (conditioningField) {
          parsed[nodeKey] = conditioningField;
        }
        return;
      }

      if ('unet' in nodeItem && 'scheduler' in nodeItem) {
        const unetField = parseUNetField(nodeItem);
        if (unetField) {
          parsed[nodeKey] = unetField;
        }
      }

      if ('tokenizer' in nodeItem && 'text_encoder' in nodeItem) {
        const clipField = parseClipField(nodeItem);
        if (clipField) {
          parsed[nodeKey] = clipField;
        }
      }

      if ('vae' in nodeItem) {
        const vaeField = parseVaeField(nodeItem);
        if (vaeField) {
          parsed[nodeKey] = vaeField;
        }
      }
    }

    // otherwise we accept any string, number or boolean
    if (isString(nodeItem) || isNumber(nodeItem) || isBoolean(nodeItem)) {
      parsed[nodeKey] = nodeItem;
      return;
    }
  });

  if (size(parsed) === 0) {
    return;
  }

  return parsed;
};

export const parseInvokeAIMetadata = (
  metadata: Record<string | number, any> | undefined
): InvokeAIMetadata | undefined => {
  if (metadata === undefined) {
    return;
  }

  if (!isObject(metadata)) {
    return;
  }

  const parsed: InvokeAIMetadata = {};

  forEach(metadata, (item, key) => {
    if (key === 'session_id' && isString(item)) {
      parsed['session_id'] = item;
    }

    if (key === 'node' && isObject(item)) {
      const nodeMetadata = parseNodeMetadata(item);

      if (nodeMetadata) {
        parsed['node'] = nodeMetadata;
      }
    }
  });

  if (size(parsed) === 0) {
    return;
  }

  return parsed;
};
