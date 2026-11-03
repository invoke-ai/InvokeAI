/** Home's dynamic edge to the Model Manager's install queue model, in its own module so tests can fail the load. */
export const importQueueModel = () => import('@features/models/ui/install-queue/queueModel');
