import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { nodeTemplatesBuilt } from 'features/nodes/store/nodesSlice';
import { parseSchema } from 'features/nodes/util/parseSchema';
import { size } from 'lodash-es';
import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { startAppListening } from '..';

export const addReceivedOpenAPISchemaListener = () => {
  startAppListening({
    actionCreator: receivedOpenAPISchema.fulfilled,
    effect: (action, { dispatch }) => {
      const log = logger('system');
      const schemaJSON = action.payload;

      log.debug({ schemaJSON }, 'Received OpenAPI schema');

      const nodeTemplates = parseSchema(schemaJSON);

      log.debug(
        { nodeTemplates: parseify(nodeTemplates) },
        `Built ${size(nodeTemplates)} node templates`
      );

      dispatch(nodeTemplatesBuilt(nodeTemplates));
    },
  });

  startAppListening({
    actionCreator: receivedOpenAPISchema.rejected,
    effect: (action) => {
      const log = logger('system');
      log.error(
        { error: parseify(action.error) },
        'Problem retrieving OpenAPI Schema'
      );
    },
  });
};
