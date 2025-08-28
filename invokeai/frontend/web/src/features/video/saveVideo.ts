import type { AppDispatch, AppGetState } from 'app/store/store';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { buildRunGraphDependencies, runGraph } from 'services/api/run-graph';
import { $socket } from 'services/events/stores';
import { assert } from 'tsafe';
import { buildSaveVideoGraph } from './graph';
import { saveVideoApi } from './state';



export const saveVideo = async (arg: { dispatch: AppDispatch; getState: AppGetState; taskId?: string }) => {
  const { dispatch, getState, taskId } = arg;
  const socket = $socket.get();
  if (!socket) {
    return;
  }
  const { graph, outputNodeId } = buildSaveVideoGraph({
    state: getState(),
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
    saveVideoApi.setSuccess(output.value);
  } catch {
    saveVideoApi.reset();
    toast({
      id: 'SAVE_VIDEO_FAILED',
      title: t('toast.saveVideoFailed'),
      status: 'error',
    });
  }
};
