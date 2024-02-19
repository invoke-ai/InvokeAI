import { FieldParseError } from 'features/nodes/types/error';
import type { FieldType } from 'features/nodes/types/field';
import type { OpenAPIV3_1SchemaOrRef } from 'features/nodes/types/openapi';
import {
  isArraySchemaObject,
  isInvocationFieldSchema,
  isNonArraySchemaObject,
  isRefObject,
  isSchemaObject,
} from 'features/nodes/types/openapi';
import { t } from 'i18next';
import { isArray } from 'lodash-es';
import type { OpenAPIV3_1 } from 'openapi-types';

/**
 * Transforms an invocation output ref object to field type.
 * @param ref The ref string to transform
 * @returns The field type.
 *
 * @example
 * refObjectToFieldType({ "$ref": "#/components/schemas/ImageField" }) --> 'ImageField'
 */
export const refObjectToSchemaName = (refObject: OpenAPIV3_1.ReferenceObject) => refObject.$ref.split('/').slice(-1)[0];

const OPENAPI_TO_FIELD_TYPE_MAP: Record<string, string> = {
  integer: 'IntegerField',
  number: 'FloatField',
  string: 'StringField',
  boolean: 'BooleanField',
};

const isCollectionFieldType = (fieldType: string) => {
  /**
   * CollectionField is `list[Any]` in the pydantic schema, but we need to distinguish between
   * it and other `list[Any]` fields, due to its special internal handling.
   *
   * In pydantic, it gets an explicit field type of `CollectionField`.
   */
  if (fieldType === 'CollectionField') {
    return true;
  }
  return false;
};

export const parseFieldType = (schemaObject: OpenAPIV3_1SchemaOrRef): FieldType => {
  if (isInvocationFieldSchema(schemaObject)) {
    // Check if this field has an explicit type provided by the node schema
    const { ui_type } = schemaObject;
    if (ui_type) {
      return {
        name: ui_type,
        isCollection: isCollectionFieldType(ui_type),
        isCollectionOrScalar: false,
      };
    }
  }
  if (isSchemaObject(schemaObject)) {
    if (schemaObject.const) {
      // Fields with a single const value are defined as `Literal["value"]` in the pydantic schema - it's actually an enum
      return {
        name: 'EnumField',
        isCollection: false,
        isCollectionOrScalar: false,
      };
    }
    if (!schemaObject.type) {
      // if schemaObject has no type, then it should have one of allOf, anyOf, oneOf

      if (schemaObject.allOf) {
        const allOf = schemaObject.allOf;
        if (allOf && allOf[0] && isRefObject(allOf[0])) {
          // This is a single ref type
          const name = refObjectToSchemaName(allOf[0]);
          if (!name) {
            throw new FieldParseError(t('nodes.unableToExtractSchemaNameFromRef'));
          }
          return {
            name,
            isCollection: false,
            isCollectionOrScalar: false,
          };
        }
      } else if (schemaObject.anyOf) {
        // ignore null types
        const filteredAnyOf = schemaObject.anyOf.filter((i) => {
          if (isSchemaObject(i)) {
            if (i.type === 'null') {
              return false;
            }
          }
          return true;
        });
        if (filteredAnyOf.length === 1) {
          // This is a single ref type
          if (isRefObject(filteredAnyOf[0])) {
            const name = refObjectToSchemaName(filteredAnyOf[0]);
            if (!name) {
              throw new FieldParseError(t('nodes.unableToExtractSchemaNameFromRef'));
            }

            return {
              name,
              isCollection: false,
              isCollectionOrScalar: false,
            };
          } else if (isSchemaObject(filteredAnyOf[0])) {
            return parseFieldType(filteredAnyOf[0]);
          }
        }
        /**
         * Handle CollectionOrScalar inputs, eg string | string[]. In OpenAPI, this is:
         * - an `anyOf` with two items
         * - one is an `ArraySchemaObject` with a single `SchemaObject or ReferenceObject` of type T in its `items`
         * - the other is a `SchemaObject` or `ReferenceObject` of type T
         *
         * Any other cases we ignore.
         */

        if (filteredAnyOf.length !== 2) {
          // This is a union of more than 2 types, which we don't support
          throw new FieldParseError(
            t('nodes.unsupportedAnyOfLength', {
              count: filteredAnyOf.length,
            })
          );
        }

        let firstType: string | undefined;
        let secondType: string | undefined;

        if (isArraySchemaObject(filteredAnyOf[0])) {
          // first is array, second is not
          const first = filteredAnyOf[0].items;
          const second = filteredAnyOf[1];
          if (isRefObject(first) && isRefObject(second)) {
            firstType = refObjectToSchemaName(first);
            secondType = refObjectToSchemaName(second);
          } else if (isNonArraySchemaObject(first) && isNonArraySchemaObject(second)) {
            firstType = first.type;
            secondType = second.type;
          }
        } else if (isArraySchemaObject(filteredAnyOf[1])) {
          // first is not array, second is
          const first = filteredAnyOf[0];
          const second = filteredAnyOf[1].items;
          if (isRefObject(first) && isRefObject(second)) {
            firstType = refObjectToSchemaName(first);
            secondType = refObjectToSchemaName(second);
          } else if (isNonArraySchemaObject(first) && isNonArraySchemaObject(second)) {
            firstType = first.type;
            secondType = second.type;
          }
        }
        if (firstType && firstType === secondType) {
          return {
            name: OPENAPI_TO_FIELD_TYPE_MAP[firstType] ?? firstType,
            isCollection: false,
            isCollectionOrScalar: true, // <-- don't forget, CollectionOrScalar type!
          };
        }

        throw new FieldParseError(
          t('nodes.unsupportedMismatchedUnion', {
            firstType,
            secondType,
          })
        );
      }
    } else if (schemaObject.enum) {
      return {
        name: 'EnumField',
        isCollection: false,
        isCollectionOrScalar: false,
      };
    } else if (schemaObject.type) {
      if (schemaObject.type === 'array') {
        // We need to get the type of the items
        if (isSchemaObject(schemaObject.items)) {
          const itemType = schemaObject.items.type;
          if (!itemType || isArray(itemType)) {
            throw new FieldParseError(
              t('nodes.unsupportedArrayItemType', {
                type: itemType,
              })
            );
          }
          // This is an OpenAPI primitive - 'null', 'object', 'array', 'integer', 'number', 'string', 'boolean'
          const name = OPENAPI_TO_FIELD_TYPE_MAP[itemType];
          if (!name) {
            // it's 'null', 'object', or 'array' - skip
            throw new FieldParseError(
              t('nodes.unsupportedArrayItemType', {
                type: itemType,
              })
            );
          }
          return {
            name,
            isCollection: true, // <-- don't forget, collection!
            isCollectionOrScalar: false,
          };
        }

        // This is a ref object, extract the type name
        const name = refObjectToSchemaName(schemaObject.items);
        if (!name) {
          throw new FieldParseError(t('nodes.unableToExtractSchemaNameFromRef'));
        }
        return {
          name,
          isCollection: true, // <-- don't forget, collection!
          isCollectionOrScalar: false,
        };
      } else if (!isArray(schemaObject.type)) {
        // This is an OpenAPI primitive - 'null', 'object', 'array', 'integer', 'number', 'string', 'boolean'
        const name = OPENAPI_TO_FIELD_TYPE_MAP[schemaObject.type];
        if (!name) {
          // it's 'null', 'object', or 'array' - skip
          throw new FieldParseError(
            t('nodes.unsupportedArrayItemType', {
              type: schemaObject.type,
            })
          );
        }
        return {
          name,
          isCollection: false,
          isCollectionOrScalar: false,
        };
      }
    }
  } else if (isRefObject(schemaObject)) {
    const name = refObjectToSchemaName(schemaObject);
    if (!name) {
      throw new FieldParseError(t('nodes.unableToExtractSchemaNameFromRef'));
    }
    return {
      name,
      isCollection: false,
      isCollectionOrScalar: false,
    };
  }
  throw new FieldParseError(t('nodes.unableToParseFieldType'));
};
