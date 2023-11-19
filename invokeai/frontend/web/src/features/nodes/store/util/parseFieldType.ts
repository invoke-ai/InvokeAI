import { FieldType } from 'features/nodes/types/types';

export const getIsPolymorphic = (type: FieldType | string): boolean =>
  type.endsWith('Polymorphic');

export const getIsCollection = (type: FieldType | string): boolean =>
  type.endsWith('Collection');

export const getBaseType = (type: FieldType | string): FieldType | string =>
  getIsPolymorphic(type)
    ? type.replace(/Polymorphic$/, '')
    : getIsCollection(type)
    ? type.replace(/Collection$/, '')
    : type;
