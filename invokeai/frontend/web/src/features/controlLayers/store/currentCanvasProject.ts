import { atom } from 'nanostores';

/**
 * Tracks the server-side project name of the canvas project most recently loaded into the
 * editor. `null` when no project is loaded — i.e. the canvas was started fresh, loaded from a
 * local `.invk` file, or the user opted to start a new canvas after saving.
 *
 * Consumed by the Save dialog to surface an "Update existing" option, and by the load flow to
 * auto-resave with remapped image names so repeated loads of the same project don't keep
 * uploading the same embedded images.
 */
export const $currentCanvasProjectName = atom<string | null>(null);
