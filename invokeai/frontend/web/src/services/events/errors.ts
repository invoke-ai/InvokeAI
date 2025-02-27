/**
 * A custom error class for queue event errors. These errors have a type, message and traceback.
 */

export class QueueError extends Error {
  type: string;
  traceback: string;

  constructor(type: string, message: string, traceback: string) {
    super(message);
    this.name = 'QueueError';
    this.type = type;
    this.traceback = traceback;

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, QueueError);
    }
  }

  toString() {
    return `${this.name} [${this.type}]: ${this.message}\nTraceback:\n${this.traceback}`;
  }
}
