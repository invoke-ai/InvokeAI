import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { SerializableObject } from 'common/types';
import { parseify } from 'common/util/serialize';
import { $templates } from 'features/nodes/store/nodesSlice';
import { parseSchema } from 'features/nodes/util/schema/parseSchema';
import { size } from 'lodash-es';
import { serializeError } from 'serialize-error';
import { appInfoApi } from 'services/api/endpoints/appInfo';

const log = logger('system');

export const addGetOpenAPISchemaListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: appInfoApi.endpoints.getOpenAPISchema.matchFulfilled,
    effect: (action, { getState }) => {
      const schemaJSON = action.payload;

      log.debug({ schemaJSON: parseify(schemaJSON) } as SerializableObject, 'Received OpenAPI schema');
      const { nodesAllowlist, nodesDenylist } = getState().config;

      const nodeTemplates = parseSchema(schemaJSON, nodesAllowlist, nodesDenylist);

      log.debug({ nodeTemplates } as SerializableObject, `Built ${size(nodeTemplates)} node templates`);

      $templates.set(nodeTemplates);
    },
  });

  startAppListening({
    matcher: appInfoApi.endpoints.getOpenAPISchema.matchRejected,
    effect: (action) => {
      // If action.meta.condition === true, the request was canceled/skipped because another request was in flight or
      // the value was already in the cache. We don't want to log these errors.
      if (!action.meta.condition) {
        log.error({ error: serializeError(action.error) }, 'Problem retrieving OpenAPI Schema');
      }
    },
  });
};
