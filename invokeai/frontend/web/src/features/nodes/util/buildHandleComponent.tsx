import { Tooltip } from '@chakra-ui/react';
import { Handle, Position } from 'reactflow';
import {
  FIELDS,
  InputField,
  OutputField,
  ProcessedNodeSchemaObject,
} from '../types';

// export const buildInputHandleComponent = (field: ProcessedNodeSchemaObject) => {
//   const color =
//     field.fieldType in FIELDS ? FIELDS[field.fieldType].color : 'gray';
//   return (
//     <Tooltip
//       key={field.title}
//       label={field.fieldType}
//       placement="start"
//       hasArrow
//     >
//       <Handle
//         type="target"
//         id={field.title}
//         position={Position.Left}
//         style={{
//           position: 'absolute',
//           left: '-1.5rem',
//           width: '1rem',
//           height: '1rem',
//           backgroundColor: `var(--invokeai-colors-${color}-500)`,
//         }}
//       />
//     </Tooltip>
//   );
// };

// export const buildOutputHandleComponent = (
//   field: ProcessedNodeSchemaObject,
//   top: string
// ) => {
//   const color =
//     field.fieldType in FIELDS ? FIELDS[field.fieldType].color : 'gray';
//   return (
//     <Tooltip key={field.title} label={field.fieldType} placement="end" hasArrow>
//       <Handle
//         type="target"
//         id={field.title}
//         position={Position.Right}
//         style={{
//           position: 'absolute',
//           top,
//           right: '-0.5rem',
//           width: '1rem',
//           height: '1rem',
//           backgroundColor: `var(--invokeai-colors-${color}-500)`,
//         }}
//       />
//     </Tooltip>
//   );
// };

export const _buildInputHandleComponent = (
  nodeId: string,
  field: InputField
) => {
  const { title, type, description } = field;
  return (
    <Tooltip
      key={type}
      label={`${title}: ${description}`}
      placement="start"
      hasArrow
    >
      <Handle
        type="target"
        id={type}
        position={Position.Left}
        style={{
          position: 'absolute',
          left: '-1.5rem',
          width: '1rem',
          height: '1rem',
          backgroundColor: `var(--invokeai-colors-${FIELDS[type].color}-500)`,
        }}
      />
    </Tooltip>
  );
};

export const _buildOutputHandleComponent = (
  nodeId: string,
  field: OutputField,
  top: string
) => {
  const { title, type, description } = field;
  return (
    <Tooltip
      key={type}
      label={`${title}: ${description}`}
      placement="end"
      hasArrow
    >
      <Handle
        type="target"
        id={type}
        position={Position.Right}
        style={{
          position: 'absolute',
          top,
          right: '-0.5rem',
          width: '1rem',
          height: '1rem',
          backgroundColor: `var(--invokeai-colors-${FIELDS[type].color}-500)`,
        }}
      />
    </Tooltip>
  );
};
