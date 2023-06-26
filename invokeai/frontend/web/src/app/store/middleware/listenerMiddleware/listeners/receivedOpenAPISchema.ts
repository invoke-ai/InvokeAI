import { receivedOpenAPISchema } from 'services/api/thunks/schema';
import { startAppListening } from '..';
import { log } from 'app/logging/useLogger';
import { parseSchema } from 'features/nodes/util/parseSchema';
import { nodeTemplatesBuilt } from 'features/nodes/store/nodesSlice';
import { size } from 'lodash-es';

const schemaLog = log.child({ namespace: 'schema' });

export const addReceivedOpenAPISchemaListener = () => {
  startAppListening({
    actionCreator: receivedOpenAPISchema.fulfilled,
    effect: (action, { dispatch, getState }) => {
      const schemaJSON = action.payload;

      schemaLog.info({ data: { schemaJSON } }, 'Dereferenced OpenAPI schema');

      const nodeTemplates = parseSchema(schemaJSON);

      schemaLog.info(
        { data: { nodeTemplates } },
        `Built ${size(nodeTemplates)} node templates`
      );

      dispatch(nodeTemplatesBuilt(nodeTemplates));
    },
  });

  startAppListening({
    actionCreator: receivedOpenAPISchema.rejected,
    effect: (action, { dispatch, getState }) => {
      schemaLog.error('Problem dereferencing OpenAPI Schema');
    },
  });
};
