/**
 * Every Launchpad panel that reads live backend state, behind one import.
 *
 * These are the only Home surfaces that need Gallery, Models, or Queue code,
 * none of which belongs in the Launchpad's initial graph. Giving each its own
 * `lazy()` boundary cost ten extra requests at first paint, because each
 * dynamic entry split its own copy of the shared feature graph out with it.
 * Collapsing them into this single module means one dynamic chunk for all
 * three, while `LivePanel` still lets the page place them independently.
 */
export { RecentOutputs } from '@features/gallery/launchpad';
export { ModelsNotice } from '@features/models/launchpad';
export { QueueStatusBand } from '@features/queue/launchpad';
