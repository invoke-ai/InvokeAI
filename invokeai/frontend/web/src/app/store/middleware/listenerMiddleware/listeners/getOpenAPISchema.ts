import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { nodeTemplatesBuilt } from 'features/nodes/store/nodesSlice';
import { parseSchema } from 'features/nodes/util/schema/parseSchema';
import { size } from 'lodash-es';
import { appInfoApi } from 'services/api/endpoints/appInfo';

import { startAppListening } from '..';

export const addGetOpenAPISchemaListener = () => {
  startAppListening({
    matcher: appInfoApi.endpoints.getOpenAPISchema.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const log = logger('system');
      const schemaJSON = action.payload;

      log.debug({ schemaJSON: parseify(schemaJSON) }, 'Received OpenAPI schema');
      const { nodesAllowlist, nodesDenylist } = getState().config;

      const nodeTemplates = parseSchema(schemaJSON, nodesAllowlist, nodesDenylist);

      log.debug({ nodeTemplates: parseify(nodeTemplates) }, `Built ${size(nodeTemplates)} node templates`);

      dispatch(nodeTemplatesBuilt(nodeTemplates));
    },
  });

  startAppListening({
    matcher: appInfoApi.endpoints.getOpenAPISchema.matchRejected,
    effect: (action) => {
      // If action.meta.condition === true, the request was canceled/skipped because another request was in flight or
      // the value was already in the cache. We don't want to log these errors.
      if (!action.meta.condition) {
        const log = logger('system');
        log.error({ error: parseify(action.error) }, 'Problem retrieving OpenAPI Schema');
      }
    },
  });
};
