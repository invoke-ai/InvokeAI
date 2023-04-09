import { filter } from 'lodash';
import { OpenAPIV3 } from 'openapi-types';
import {
  isReferenceObject,
  NodeSchemaObject,
  NodesComponentsObject,
  ProcessedNodeSchemaObject,
  _isReferenceObject,
  _isSchemaObject,
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

export const _parseOutputRef = (
  components: OpenAPIV3.ComponentsObject,
  ref: OpenAPIV3.ReferenceObject
): string[] => {
  // extract output schema name from ref
  const outputSchemaName = ref.$ref.split('/').slice(-1)[0];

  // TODO: recursively parse refs? currently just manually going one level deep
  const outputSchema = components.schemas![outputSchemaName];

  const filteredProperties = filter(
    (outputSchema as OpenAPIV3.SchemaObject).properties,
    (prop, key) => key !== 'type'
  );

  const outputFieldTypes: string[] = [];

  filteredProperties.forEach((property) => {
    if (_isSchemaObject(property)) {
      if (property.allOf) {
        outputFieldTypes.push(
          getFieldTypeFromRefString(
            (property.allOf[0] as OpenAPIV3.ReferenceObject).$ref
          )
        );
      } else {
        outputFieldTypes.push(property.type!.replace('number', 'float'));
      }
    }
  });

  return outputFieldTypes;
};
