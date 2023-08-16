import { SystemStyleObject } from '@chakra-ui/styled-system';

const selectionStyles: SystemStyleObject = {
  backgroundColor: 'accentAlpha.150 !important',
  borderColor: 'accentAlpha.700 !important',
  borderRadius: 'base !important',
  borderStyle: 'dashed !important',
  _dark: {
    borderColor: 'accent.400 !important',
  },
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
