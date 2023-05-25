import { v4 as uuidv4 } from 'uuid';
import { RootState } from 'app/store/store';
import { CompelInvocation } from 'services/api';
import { O } from 'ts-toolbelt';

export const buildCompelNode = (
  prompt: string,
  state: RootState,
  overrides: O.Partial<CompelInvocation, 'deep'> = {}
): CompelInvocation => {
  const nodeId = uuidv4();
  const { generation } = state;

  const { model } = generation;

  const compelNode: CompelInvocation = {
    id: nodeId,
    type: 'compel',
    prompt,
    model,
  };

  Object.assign(compelNode, overrides);

  return compelNode;
};
