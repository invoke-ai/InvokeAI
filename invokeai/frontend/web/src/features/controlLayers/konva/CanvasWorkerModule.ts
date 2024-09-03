import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Extents, ExtentsResult, GetBboxTask, WorkerLogMessage } from 'features/controlLayers/konva/worker';
import type { Logger } from 'roarr';

export class CanvasWorkerModule extends CanvasModuleBase {
  readonly type = 'worker';
  readonly id: string;
  readonly path: string[];
  readonly log: Logger;
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;

  worker: Worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'worker' });
  tasks: Map<string, { task: GetBboxTask; onComplete: (extents: Extents | null) => void }> = new Map();

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.worker.onmessage = (event: MessageEvent<ExtentsResult | WorkerLogMessage>) => {
      const { type, data } = event.data;
      if (type === 'log') {
        if (data.ctx) {
          this.log[data.level](data.ctx, data.message);
        } else {
          this.log[data.level](data.message);
        }
      } else if (type === 'extents') {
        const task = this.tasks.get(data.id);
        if (!task) {
          return;
        }
        task.onComplete(data.extents);
        this.tasks.delete(data.id);
      }
    };
    this.worker.onerror = (event) => {
      this.log.error({ message: event.message }, 'Worker error');
    };
    this.worker.onmessageerror = () => {
      this.log.error('Worker message error');
    };
  }

  requestBbox(data: Omit<GetBboxTask['data'], 'id'>, onComplete: (extents: Extents | null) => void) {
    const id = getPrefixedId('bbox_calculation');
    const task: GetBboxTask = {
      type: 'get_bbox',
      data: { ...data, id },
    };
    this.tasks.set(id, { task, onComplete });
    this.worker.postMessage(task, [data.buffer]);
  }

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      tasks: Array.from(this.tasks.keys()),
    };
  };

  destroy = () => {
    this.log.trace('Destroying worker module');
    this.worker.terminate();
    this.tasks.clear();
  };
}
