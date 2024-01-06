import type { SystemStyleObject } from '@chakra-ui/styled-system';

const selectionStyles: SystemStyleObject = {
  backgroundColor: 'blueAlpha.150 !important',
  borderColor: 'blue.400 !important',
  borderRadius: 'base !important',
  borderStyle: 'dashed !important',
};

export const reactflowStyles: SystemStyleObject = {
  '.react-flow__nodesselection-rect': {
    ...selectionStyles,
    padding: '1rem !important',
    boxSizing: 'content-box !important',
    transform: 'translate(-1rem, -1rem) !important',
  },
  '.react-flow__selection': selectionStyles,
};
