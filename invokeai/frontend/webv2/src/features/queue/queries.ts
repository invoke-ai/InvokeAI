/**
 * Published Queue query surface for shell consumers (e.g. the command
 * palette's queue search). Widget-internal data hooks stay private.
 */
export { getQueueReadModelOptions } from './publicApi';
export { getQueueQueryScope, type QueueJobsScope } from './ui/queueScope';
