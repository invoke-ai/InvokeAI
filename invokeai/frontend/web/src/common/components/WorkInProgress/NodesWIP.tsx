import WorkInProgress from './WorkInProgress';
import ReactFlow, {
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  Controls,
  Edge,
  Node,
  NodeTypes,
  OnEdgesChange,
  OnNodesChange,
} from 'reactflow';

import 'reactflow/dist/style.css';
import {
  FunctionComponent,
  ReactNode,
  useCallback,
  useMemo,
  useState,
} from 'react';
import { OpenAPIV3 } from 'openapi-types';
import { filter, map } from 'lodash';
import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  Input,
  Select,
  Switch,
  Text,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Tooltip,
} from '@chakra-ui/react';

// grab the openapi schema json
async function fetchOpenAPI(): Promise<OpenAPIV3.Document> {
  const response = await fetch(`openapi.json`);
  const jsonData = await response.json();
  return jsonData;
}

// build an individual input element based on the schema
const buildInputElement = (field: OpenAPIV3.SchemaObject): ReactNode => {
  if (field.type === 'string') {
    // `string` fields may either be a text input or an enum ie select
    if (field.enum) {
      return (
        <Select defaultValue={field.default}>
          {field.enum?.map((option) => (
            <option key={option}>{option}</option>
          ))}
        </Select>
      );
    }

    return <Input defaultValue={field.default}></Input>;
  } else if (field.type === 'boolean') {
    return <Switch defaultValue={field.default}></Switch>;
  } else if (['integer', 'number'].includes(field.type as string)) {
    return (
      <NumberInput defaultValue={field.default}>
        <NumberInputField />
        <NumberInputStepper>
          <NumberIncrementStepper />
          <NumberDecrementStepper />
        </NumberInputStepper>
      </NumberInput>
    );
  }
};

// build object of invocations UI components keyed by their name
const buildInvocations = async () => {
  // get schema
  const openApi = await fetchOpenAPI();

  // filter out non-invocation schemas, kinda janky but dunno if there's another way really
  // outputs an array
  const filteredSchemas = filter(
    openApi.components?.schemas,
    (_schema, key) =>
      key.includes('Invocation') && !key.includes('InvocationOutput')
  );

  // actually build the UI components
  // reduce the array of schemas into an object of react function components, keyed by name (eg NodeTypes)
  const invocations = filteredSchemas.reduce<NodeTypes>((acc, val, key) => {
    // we know these are always SchemaObjects not ReferenceObjects
    const schema = val as OpenAPIV3.SchemaObject;

    // we know `title` will always be present
    const name = schema.title!.replace('Invocation', '');

    // `type` and `id` are not valid inputs/outputs
    const fields = filter(
      schema.properties,
      (prop, key) => !['type', 'id'].includes(key)
    );

    // assemble!
    acc[name] = () => (
      <Box sx={{ padding: 4, bg: 'base.800', borderRadius: 'md' }}>
        <Flex flexDirection="column">
          <Text>{name}</Text>
          {fields.map((field, i) => {
            const f = field as OpenAPIV3.SchemaObject;
            return (
              <Tooltip key={i} label={f.description} placement="top" hasArrow>
                <FormControl>
                  <FormLabel>{f.title}</FormLabel>
                  {buildInputElement(f)}
                </FormControl>
              </Tooltip>
            );
          })}
        </Flex>
      </Box>
    );

    return acc;
  }, {});

  console.log(invocations);

  return invocations;
};

const invocations = await buildInvocations();

// make initial nodes one of every node for now
let n = 0;
const initialNodes = map(
  invocations,
  (i, type): Node => ({
    id: type,
    type,
    position: { x: (n += 20), y: (n += 20) },
    data: {},
  })
);

export default function NodesWIP() {
  const nodeTypes = useMemo(() => invocations, []);
  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>([]);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setEdges((eds: Edge[]) => applyEdgeChanges(changes, eds)),
    []
  );
  return (
    <WorkInProgress>
      <ReactFlow
        nodeTypes={nodeTypes}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
      >
        <Background />
        <Controls />
      </ReactFlow>
    </WorkInProgress>
  );
}
