import type {
  FloatFieldCollectionInputTemplate,
  FloatFieldCollectionValue,
  ImageFieldCollectionInputTemplate,
  ImageFieldCollectionValue,
  IntegerFieldCollectionInputTemplate,
  IntegerFieldCollectionValue,
  StringFieldCollectionInputTemplate,
  StringFieldCollectionValue,
} from 'features/nodes/types/field';
import { t } from 'i18next';

export const validateImageFieldCollectionValue = (
  value: NonNullable<ImageFieldCollectionValue>,
  template: ImageFieldCollectionInputTemplate
): string[] => {
  const reasons: string[] = [];
  const { minItems, maxItems } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  }

  if (minItems !== undefined && count < minItems) {
    reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
  }

  if (maxItems !== undefined && count > maxItems) {
    reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
  }

  return reasons;
};

export const validateStringFieldCollectionValue = (
  value: NonNullable<StringFieldCollectionValue>,
  template: StringFieldCollectionInputTemplate
): string[] => {
  const reasons: string[] = [];
  const { minItems, maxItems, minLength, maxLength } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  }

  if (minItems !== undefined && count < minItems) {
    reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
  }

  if (maxItems !== undefined && count > maxItems) {
    reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
  }

  for (const str of value) {
    if (maxLength !== undefined && str.length > maxLength) {
      reasons.push(t('parameters.invoke.collectionStringTooLong', { value, maxLength }));
    }
    if (minLength !== undefined && str.length < minLength) {
      reasons.push(t('parameters.invoke.collectionStringTooShort', { value, minLength }));
    }
  }

  return reasons;
};

export const validateNumberFieldCollectionValue = (
  value: NonNullable<IntegerFieldCollectionValue> | NonNullable<FloatFieldCollectionValue>,
  template: IntegerFieldCollectionInputTemplate | FloatFieldCollectionInputTemplate
): string[] => {
  const reasons: string[] = [];
  const { minItems, maxItems, minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  }

  if (minItems !== undefined && count < minItems) {
    reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
  }

  if (maxItems !== undefined && count > maxItems) {
    reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
  }

  for (const num of value) {
    if (maximum !== undefined && num > maximum) {
      reasons.push(t('parameters.invoke.collectionNumberGTMax', { value, maximum }));
    }
    if (minimum !== undefined && num < minimum) {
      reasons.push(t('parameters.invoke.collectionNumberLTMin', { value, minimum }));
    }
    if (exclusiveMaximum !== undefined && num >= exclusiveMaximum) {
      reasons.push(t('parameters.invoke.collectionNumberGTExclusiveMax', { value, exclusiveMaximum }));
    }
    if (exclusiveMinimum !== undefined && num <= exclusiveMinimum) {
      reasons.push(t('parameters.invoke.collectionNumberLTExclusiveMin', { value, exclusiveMinimum }));
    }
    if (multipleOf !== undefined && num % multipleOf !== 0) {
      reasons.push(t('parameters.invoke.collectionNumberNotMultipleOf', { value, multipleOf }));
    }
  }

  return reasons;
};
