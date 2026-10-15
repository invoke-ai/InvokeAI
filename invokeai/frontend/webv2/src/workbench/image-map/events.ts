/** Image-map Socket.IO contracts mirror snake_case models in events_common.py and remain owned by this domain. */

/** Embedding-index progress counts. Routed to admins only by the backend. */
export interface ImageIndexStatusEvent {
  total: number;
  embedded: number;
  pending: number;
  /**
   * Failed embeddings are excluded from pending, explaining settled totals above embedded counts. Optional for
   * older servers; default zero.
   */
  failed?: number;
}

/** User-room refresh signal after embedding; nonadmins need it because admin index status never reaches them. */
export interface ImageIndexUpdatedEvent {
  user_id: string;
}

/** A user's image map projection finished recomputing (user + admin rooms). */
export interface ImageMapProjectionReadyEvent {
  user_id: string;
  point_count: number;
}
