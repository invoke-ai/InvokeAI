import { logger } from 'app/logging/logger';
import { userInvoked } from 'app/store/actions';
import { parseify } from 'common/util/serialize';
import { nodesGraphBuilt } from 'features/nodes/store/actions';
import { buildNodesGraph } from 'features/nodes/util/graphBuilders/buildNodesGraph';
import { sessionReadyToInvoke } from 'features/system/store/actions';
import { sessionCreated } from 'services/api/thunks/session';
import { startAppListening } from '..';

export const addUserInvokedNodesListener = () => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof userInvoked> =>
      userInvoked.match(action) && action.payload === 'nodes',
    effect: async (action, { getState, dispatch, take }) => {
      const log = logger('session');
      const state = getState();

      const graph = buildNodesGraph(state.nodes);
      dispatch(nodesGraphBuilt(graph));
      log.debug({ graph: parseify(graph) }, 'Nodes graph built');

      dispatch(sessionCreated({ graph }));

      await take(sessionCreated.fulfilled.match);

      dispatch(sessionReadyToInvoke());
    },
  });
};
