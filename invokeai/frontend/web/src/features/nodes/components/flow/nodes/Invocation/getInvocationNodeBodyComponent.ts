export type InvocationNodeBodyComponentKey = 'default' | 'call_saved_workflows';

export const getInvocationNodeBodyComponentKey = (type: string): InvocationNodeBodyComponentKey => {
  if (type === 'call_saved_workflows') {
    return 'call_saved_workflows';
  }

  return 'default';
};
