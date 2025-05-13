import { zAspectRatioID, zImageWithDims } from 'features/controlLayers/store/types';
import { zParameterModel } from 'features/parameters/types/parameterSchemas';
import { z } from 'zod';

const zLowMedHigh = z.enum(['low', 'med', 'high']);
const zControlType = z.enum(['line', 'depth']);

const STARTING_IMAGE_TYPE = 'starting_image';
const zStartingImage = z.object({
  type: z.literal(STARTING_IMAGE_TYPE).default(STARTING_IMAGE_TYPE),
  image: zImageWithDims.nullable().default(null),
  variation: zLowMedHigh.default('med'),
});
export type StartingImage = z.infer<typeof zStartingImage>;
export const getStartingImage = (overrides: Partial<Omit<StartingImage, 'type'>>) => zStartingImage.parse(overrides);

const REFERENCE_IMAGE_TYPE = 'reference_image';
const zReferenceImage = z.object({
  type: z.literal(REFERENCE_IMAGE_TYPE).default(REFERENCE_IMAGE_TYPE),
  image: zImageWithDims.nullable().default(null),
});
export type ReferenceImage = z.infer<typeof zReferenceImage>;
export const getReferenceImage = (overrides: Partial<Omit<ReferenceImage, 'type'>>) => zReferenceImage.parse(overrides);

const CONTROL_IMAGE_TYPE = 'control_image';
const zControlImage = z.object({
  type: z.literal(CONTROL_IMAGE_TYPE).default(CONTROL_IMAGE_TYPE),
  control_type: zControlType.default('line'),
  image: zImageWithDims.nullable().default(null),
});
export type ControlImage = z.infer<typeof zControlImage>;
export const getControlImage = (overrides: Partial<Omit<ControlImage, 'type'>>) => zControlImage.parse(overrides);

export const zSimpleGenerationState = z.object({
  _version: z.literal(1).default(1),
  positivePrompt: z.string().default(''),
  negativePrompt: z.string().default(''),
  model: zParameterModel.nullable().default(null),
  aspectRatio: zAspectRatioID.default('1:1'),
  startingImage: zStartingImage.nullable().default(null),
  referenceImages: z.array(zReferenceImage).default(() => []),
  controlImage: zControlImage.nullable().default(null),
});

export type SimpleGenerationState = z.infer<typeof zSimpleGenerationState>;
