// import WorkInProgress from './WorkInProgress';
// import ReactFlow, {
//   applyEdgeChanges,
//   applyNodeChanges,
//   Background,
//   Controls,
//   Edge,
//   Handle,
//   Node,
//   NodeTypes,
//   OnEdgesChange,
//   OnNodesChange,
//   Position,
// } from 'reactflow';

// import 'reactflow/dist/style.css';
// import {
//   Fragment,
//   FunctionComponent,
//   ReactNode,
//   useCallback,
//   useMemo,
//   useState,
// } from 'react';
// import { OpenAPIV3 } from 'openapi-types';
// import { filter, map, reduce } from 'lodash';
// import {
//   Box,
//   Flex,
//   FormControl,
//   FormLabel,
//   Input,
//   Select,
//   Switch,
//   Text,
//   NumberInput,
//   NumberInputField,
//   NumberInputStepper,
//   NumberIncrementStepper,
//   NumberDecrementStepper,
//   Tooltip,
//   chakra,
//   Badge,
//   Heading,
//   VStack,
//   HStack,
//   Menu,
//   MenuButton,
//   MenuList,
//   MenuItem,
//   MenuItemOption,
//   MenuGroup,
//   MenuOptionGroup,
//   MenuDivider,
//   IconButton,
// } from '@chakra-ui/react';
// import { FaPlus } from 'react-icons/fa';
// import {
//   FIELD_NAMES as FIELD_NAMES,
//   FIELDS,
//   INVOCATION_NAMES as INVOCATION_NAMES,
//   INVOCATIONS,
// } from 'features/nodeEditor/constants';

// console.log('invocations', INVOCATIONS);

// const nodeTypes = reduce(
//   INVOCATIONS,
//   (acc, val, key) => {
//     acc[key] = val.component;
//     return acc;
//   },
//   {} as NodeTypes
// );

// console.log('nodeTypes', nodeTypes);

// // make initial nodes one of every node for now
// let n = 0;
// const initialNodes = map(INVOCATIONS, (i) => ({
//   id: i.type,
//   type: i.title,
//   position: { x: (n += 20), y: (n += 20) },
//   data: {},
// }));

// console.log('initialNodes', initialNodes);

// export default function NodesWIP() {
//   const [nodes, setNodes] = useState<Node[]>([]);
//   const [edges, setEdges] = useState<Edge[]>([]);

//   const onNodesChange: OnNodesChange = useCallback(
//     (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
//     []
//   );

//   const onEdgesChange: OnEdgesChange = useCallback(
//     (changes) => setEdges((eds: Edge[]) => applyEdgeChanges(changes, eds)),
//     []
//   );

//   return (
//     <Box
//       sx={{
//         position: 'relative',
//         width: 'full',
//         height: 'full',
//         borderRadius: 'md',
//       }}
//     >
//       <ReactFlow
//         nodeTypes={nodeTypes}
//         nodes={nodes}
//         edges={edges}
//         onNodesChange={onNodesChange}
//         onEdgesChange={onEdgesChange}
//       >
//         <Background />
//         <Controls />
//       </ReactFlow>
//       <HStack sx={{ position: 'absolute', top: 2, right: 2 }}>
//         {FIELD_NAMES.map((field) => (
//           <Badge
//             key={field}
//             colorScheme={FIELDS[field].color}
//             sx={{ userSelect: 'none' }}
//           >
//             {field}
//           </Badge>
//         ))}
//       </HStack>
//       <Menu>
//         <MenuButton
//           as={IconButton}
//           aria-label="Options"
//           icon={<FaPlus />}
//           sx={{ position: 'absolute', top: 2, left: 2 }}
//         />
//         <MenuList>
//           {INVOCATION_NAMES.map((name) => {
//             const invocation = INVOCATIONS[name];
//             return (
//               <Tooltip
//                 key={name}
//                 label={invocation.description}
//                 placement="end"
//                 hasArrow
//               >
//                 <MenuItem>{invocation.title}</MenuItem>
//               </Tooltip>
//             );
//           })}
//         </MenuList>
//       </Menu>
//     </Box>
//   );
// }

export default {};
