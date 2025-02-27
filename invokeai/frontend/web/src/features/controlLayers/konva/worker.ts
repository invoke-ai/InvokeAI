import type { LogLevel } from 'app/logging/logger';
import type { JsonObject } from 'roarr/dist/types';

export type Extents = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

/**
 * Get the bounding box of an image.
 * @param buffer The ArrayBuffer of the image to get the bounding box of.
 * @param width The width of the image.
 * @param height The height of the image.
 * @returns The minimum and maximum x and y values of the image's bounding box, or null if the image has no pixels.
 */
const getImageDataBboxArrayBuffer = (buffer: ArrayBuffer, width: number, height: number): Extents | null => {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let alpha = 0;
  let isEmpty = true;
  const arr = new Uint8ClampedArray(buffer);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      alpha = arr[(y * width + x) * 4 + 3] ?? 0;
      if (alpha > 0) {
        isEmpty = false;
        if (x < minX) {
          minX = x;
        }
        if (x > maxX) {
          maxX = x;
        }
        if (y < minY) {
          minY = y;
        }
        if (y > maxY) {
          maxY = y;
        }
      }
    }
  }

  return isEmpty ? null : { minX, minY, maxX: maxX + 1, maxY: maxY + 1 };
};

export type GetBboxTask = {
  type: 'get_bbox';
  data: { id: string; buffer: ArrayBuffer; width: number; height: number };
};

type TaskWithTimestamps<T extends Record<string, unknown>> = T & { started: number | null; finished: number | null };

export type ExtentsResult = {
  type: 'extents';
  data: { id: string; extents: Extents | null };
};

export type WorkerLogMessage = {
  type: 'log';
  data: { level: LogLevel; message: string; ctx?: JsonObject };
};

// A single worker is used to process tasks in a queue
const queue: TaskWithTimestamps<GetBboxTask>[] = [];
let currentTask: TaskWithTimestamps<GetBboxTask> | null = null;

function postLogMessage(level: LogLevel, message: string, ctx?: JsonObject) {
  const data: WorkerLogMessage = {
    type: 'log',
    data: { level, message, ctx },
  };
  self.postMessage(data);
}

function processNextTask() {
  // Grab the next task
  const task = queue.shift();
  if (!task) {
    // Queue empty - we can clear the current task to allow the worker to resume the queue when another task is posted
    currentTask = null;
    return;
  }

  postLogMessage('debug', 'Processing task', { type: task.type, id: task.data.id });
  task.started = performance.now();

  // Set the current task so we don't process another one
  currentTask = task;

  // Process the task
  if (task.type === 'get_bbox') {
    const { buffer, width, height, id } = task.data;
    const extents = getImageDataBboxArrayBuffer(buffer, width, height);
    const result: ExtentsResult = {
      type: 'extents',
      data: { id, extents },
    };
    task.finished = performance.now();
    postLogMessage('debug', 'Task complete', {
      type: task.type,
      id: task.data.id,
      started: task.started,
      finished: task.finished,
      durationMs: task.finished - task.started,
    });
    self.postMessage(result);
  } else {
    postLogMessage('error', 'Unknown task type', { type: task.type });
  }

  // Repeat
  processNextTask();
}

self.onmessage = (event: MessageEvent<Omit<GetBboxTask, 'started' | 'finished'>>) => {
  const task = event.data;

  postLogMessage('debug', 'Received task', { type: task.type, id: task.data.id });
  // Add the task to the queue
  queue.push({ ...event.data, started: null, finished: null });

  // If we are not currently processing a task, process the next one
  if (!currentTask) {
    processNextTask();
  }
};
