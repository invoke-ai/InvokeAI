/**
 * Client-generated project ids. The backend treats `(user_id, project_id)` as
 * the primary key and never mints ids itself, so uniqueness only has to hold
 * per user; the timestamp+entropy shape keeps ids unique across devices and
 * across deleted-then-recreated projects.
 */
export const createProjectId = (): string =>
  `project-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
