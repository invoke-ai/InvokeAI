/**
 * Typed contracts for the backend Socket.IO events the image map consumes.
 * Payload shapes mirror the pydantic models in
 * `invokeai/app/services/events/events_common.py` (serialized as snake_case).
 * These events belong to the image-map domain, so they live here rather than
 * in a feature's event map.
 */

/** Embedding-index progress counts. Routed to admins only by the backend. */
export interface ImageIndexStatusEvent {
  total: number;
  embedded: number;
  pending: number;
}

/**
 * Counts-free poke to one user whose images were just (re)embedded, routed
 * to that user's room only. This is what lets a non-admin's map refresh
 * after their own generations: the status event above never reaches them.
 */
export interface ImageIndexUpdatedEvent {
  user_id: string;
}

/** A user's image map projection finished recomputing (user + admin rooms). */
export interface ImageMapProjectionReadyEvent {
  user_id: string;
  point_count: number;
}
