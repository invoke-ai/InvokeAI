import CurrentImageNode from './nodes/CurrentImageNode';
import InvocationNodeWrapper from './nodes/InvocationNodeWrapper';
import NotesNode from './nodes/NotesNode';

export const nodeTypes = {
  invocation: InvocationNodeWrapper,
  current_image: CurrentImageNode,
  notes: NotesNode,
};
