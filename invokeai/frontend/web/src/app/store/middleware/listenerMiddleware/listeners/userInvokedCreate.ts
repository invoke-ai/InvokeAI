import { startAppListening } from '..';
import { buildLinearGraph } from 'features/nodes/util/buildLinearGraph';
import { sessionCreated } from 'services/thunks/session';
import { log } from 'app/logging/useLogger';
import { createGraphBuilt } from 'features/nodes/store/actions';
import { userInvoked } from 'app/store/actions';

const moduleLog = log.child({ namespace: 'invoke' });

export const addUserInvokedCreateListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof userInvoked> =>
      userInvoked.match(action) && action.payload === 'generate',
    effect: (action, { getState, dispatch }) => {
      const state = getState();

      const graph = buildLinearGraph(state);
      dispatch(createGraphBuilt(graph));
      moduleLog({ data: graph }, 'Create graph built');

      dispatch(sessionCreated({ graph }));
    },
  });
};
