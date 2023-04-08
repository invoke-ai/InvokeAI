import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  HStack,
  Tooltip,
  Icon,
} from '@chakra-ui/react';
import { filter, uniq } from 'lodash';
import { FaInfoCircle } from 'react-icons/fa';
// import { PRIMITIVE_FIELDS } from '../constants';
import {
  Invocations,
  isNodeSchemaObject,
  isReferenceObject,
  NodesOpenAPIDocument,
  ProcessedNodeSchemaObject,
} from '../types';
import { buildFieldComponent } from './buildFieldComponent';
import {
  buildInputHandleComponent,
  buildOutputHandleComponent,
} from './buildHandleComponent';
import { fetchOpenAPISchema } from './fetchOpenAPISchema';
import { parseOutputRef } from './parseRef';

// build object of invocations UI components keyed by their name
export const buildInvocations = async (): Promise<{
  invocations: Invocations;
  fieldTypes: string[];
}> => {
  // get schema - cast as the modified OpenAPI document type
  const openApi = (await fetchOpenAPISchema()) as NodesOpenAPIDocument;

  // filter out non-invocation schemas, kinda janky but dunno if there's another way really
  // also filter out some tricky ones for now
  const filteredSchemas = filter(
    openApi.components.schemas,
    (_schema, key) =>
      key.includes('Invocation') &&
      !key.includes('InvocationOutput') &&
      !key.includes('Collect') &&
      !key.includes('Range') &&
      !key.includes('Iterate') &&
      !key.includes('LoadImage') &&
      !key.includes('Graph')
  );

  let fieldTypes: string[] = [];

  // actually build the UI components
  // reduce the array of schemas into an object of react function components, keyed by name (eg NodeTypes)
  const invocations = filteredSchemas.reduce<Invocations>(
    (acc, schema, key) => {
      // only want SchemaObjects
      if (isReferenceObject(schema)) {
        return acc;
      }

      const title = schema.title.replace('Invocation', '');

      const type = schema.properties.type.default;

      // `type` and `id` are not valid inputs/outputs
      const inputs = filter(
        schema.properties,
        (prop, key) => !['type', 'id'].includes(key) && isNodeSchemaObject(prop)
      ) as ProcessedNodeSchemaObject[]; // if i don't cast as, the type is never[], dunno why

      inputs.forEach((input) => {
        if (input.allOf && isReferenceObject(input.allOf[0])) {
          input.fieldType = input.allOf[0].$ref
            .split('/')
            .slice(-1)[0]
            .toLowerCase()
            .replace('field', ''); // ImageField --> image
        } else {
          input.fieldType = input.type.toLowerCase().replace('number', 'float');
        }

        fieldTypes.push(input.fieldType);
      });

      const outputs = [parseOutputRef(openApi.components, schema.output.$ref)];

      outputs.forEach(({ fieldType }) => {
        fieldTypes.push(fieldType);
      });

      // assemble!
      acc[title] = {
        title,
        type,
        schema,
        outputs,
        inputs,
        description: schema.description || '',
        component: () => (
          <Box
            sx={{
              padding: 4,
              bg: 'base.800',
              borderRadius: 'md',
              boxShadow: 'dark-lg',
            }}
          >
            <Flex flexDirection="column" gap={2}>
              <HStack justifyContent="space-between">
                <Heading size="sm" fontWeight={500} color="base.100">
                  {title}
                </Heading>
                <Tooltip
                  label={schema.description}
                  placement="top"
                  hasArrow
                  shouldWrapChildren
                >
                  <Icon color="base.300" as={FaInfoCircle} />
                </Tooltip>
              </HStack>
              {inputs.map((input, i) => {
                if (isNodeSchemaObject(input)) {
                  return (
                    <Box
                      key={i}
                      position="relative"
                      p={2}
                      borderWidth={1}
                      borderRadius="md"
                    >
                      <FormControl>
                        <HStack
                          justifyContent="space-between"
                          alignItems="center"
                        >
                          <FormLabel>{input.title}</FormLabel>
                          <Tooltip
                            label={input.description}
                            placement="top"
                            hasArrow
                            shouldWrapChildren
                          >
                            <Icon color="base.400" as={FaInfoCircle} />
                          </Tooltip>
                        </HStack>
                        {buildFieldComponent(input)}
                      </FormControl>
                      {buildInputHandleComponent(input)}
                    </Box>
                  );
                }
              })}
            </Flex>
            {outputs.map((output, i) => {
              const top = `${(100 / (outputs.length + 1)) * (i + 1)}%`;
              return buildOutputHandleComponent(output, top);
            })}
          </Box>
        ),
      };

      return acc;
    },
    {}
  );

  fieldTypes = uniq(fieldTypes);

  return { invocations, fieldTypes };
};
