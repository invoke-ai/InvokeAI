import { filter } from 'lodash';
import {
  isReferenceObject,
  NodeSchemaObject,
  NodesComponentsObject,
  ProcessedNodeSchemaObject,
} from '../types';

export const parseOutputRef = (
  components: NodesComponentsObject,
  ref: string
) => {
  // extract output schema name from ref
  const outputSchemaName = ref.split('/').slice(-1)[0].toString();

  // TODO: recursively parse refs? currently just manually going one level deep
  const output = components.schemas[
    outputSchemaName
  ] as unknown as ProcessedNodeSchemaObject;

  const filteredProperties = filter(
    output.properties,
    (prop, key) => key !== 'type'
  ) as NodeSchemaObject[];

  if (filteredProperties[0]?.allOf?.length) {
    if (isReferenceObject(filteredProperties[0].allOf[0])) {
      output.fieldType = filteredProperties[0].allOf[0].$ref
        .split('/')
        .slice(-1)[0]
        .toString()
        .toLowerCase()
        .replace('field', '');
    }
  } else {
    output.fieldType = filteredProperties[0].type.replace('number', 'float');
  }

  return output;
};
