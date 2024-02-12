import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { nodeTemplatesBuilt } from 'features/nodes/store/nodeTemplatesSlice';
import { parseSchema } from 'features/nodes/util/schema/parseSchema';
import { size } from 'lodash-es';
import { utilitiesApi } from 'services/api/endpoints/utilities';

import { startAppListening } from '..';

export const schemaLoadedListener = () => {
  startAppListening({
    matcher: utilitiesApi.endpoints.loadSchema.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const log = logger('system');
      const schemaJSON = action.payload;
      console.log({ action });

      log.info({ schemaJSON }, 'Received OpenAPI schema');
      const { nodesAllowlist, nodesDenylist } = getState().config;

      const nodeTemplates = parseSchema(schemaJSON, nodesAllowlist, nodesDenylist);

      log.info({ nodeTemplates: parseify(nodeTemplates) }, `Built ${size(nodeTemplates)} node templates`);

      dispatch(nodeTemplatesBuilt(nodeTemplates));
    },
  });

  startAppListening({
    matcher: utilitiesApi.endpoints.loadSchema.matchRejected,
    effect: (action) => {
      const log = logger('system');
      log.error({ error: parseify(action.error) }, 'Problem retrieving OpenAPI Schema');
    },
  });
};
