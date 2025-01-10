import type {
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

export const validateIntegerFieldCollectionValue = (
  value: NonNullable<IntegerFieldCollectionValue>,
  template: IntegerFieldCollectionInputTemplate
): string[] => {
  const reasons: string[] = [];
  const { minItems, maxItems, minimum, maximum, exclusiveMinimum, exclusiveMaximum } = template;
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

  for (const int of value) {
    if (maximum !== undefined && int > maximum) {
      reasons.push(t('parameters.invoke.collectionNumberGTMax', { value, maximum }));
    }
    if (minimum !== undefined && int < minimum) {
      reasons.push(t('parameters.invoke.collectionNumberLTMin', { value, minimum }));
    }
    if (exclusiveMaximum !== undefined && int >= exclusiveMaximum) {
      reasons.push(t('parameters.invoke.collectionNumberGTExclusiveMax', { value, exclusiveMaximum }));
    }
    if (exclusiveMinimum !== undefined && int <= exclusiveMinimum) {
      reasons.push(t('parameters.invoke.collectionNumberLTExclusiveMin', { value, exclusiveMinimum }));
    }
  }

  return reasons;
};
