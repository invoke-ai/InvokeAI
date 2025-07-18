import type { AppDispatch, AppGetState } from 'app/store/store';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { buildRunGraphDependencies, runGraph } from 'services/api/run-graph';
import type { ImageDTO } from 'services/api/types';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';

import { buildPromptExpansionGraph } from './graph';
import { promptExpansionApi } from './state';

export const expandPrompt = async (arg: { dispatch: AppDispatch; getState: AppGetState; imageDTO?: ImageDTO }) => {
  const { dispatch, getState, imageDTO } = arg;
  const socket = $socket.get();
  if (!socket) {
    return;
  }
  const { graph, outputNodeId } = buildPromptExpansionGraph({
    state: getState(),
    imageDTO,
  });
  const dependencies = buildRunGraphDependencies(dispatch, socket);
  try {
    const { output } = await runGraph({
      graph,
      outputNodeId,
      dependencies,
      options: {
        prepend: true,
      },
    });
    assert(output.type === 'string_output');
    promptExpansionApi.setSuccess(output.value);
  } catch {
    promptExpansionApi.reset();
    toast({
      id: 'PROMPT_EXPANSION_FAILED',
      title: t('toast.promptExpansionFailed'),
      status: 'error',
    });
  }
};
