import { startAppListening } from '..';
import { sessionCreated } from 'services/thunks/session';
import { buildNodesGraph } from 'features/nodes/util/graphBuilders/buildNodesGraph';
import { log } from 'app/logging/useLogger';
import { nodesGraphBuilt } from 'features/nodes/store/actions';
import { userInvoked } from 'app/store/actions';
import { sessionReadyToInvoke } from 'features/system/store/actions';

const moduleLog = log.child({ namespace: 'invoke' });

export const addUserInvokedNodesListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof userInvoked> =>
      userInvoked.match(action) && action.payload === 'nodes',
    effect: async (action, { getState, dispatch, take }) => {
      const state = getState();

      const graph = buildNodesGraph(state);
      dispatch(nodesGraphBuilt(graph));
      moduleLog.debug({ data: graph }, 'Nodes graph built');

      dispatch(sessionCreated({ graph }));

      await take(sessionCreated.fulfilled.match);

      dispatch(sessionReadyToInvoke());
    },
  });
};
