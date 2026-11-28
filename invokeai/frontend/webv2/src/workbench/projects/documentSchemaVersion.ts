/** The whole-document schema the browser works in, and a project the server holds whole is saved in. */
export const PROJECT_DOCUMENT_SCHEMA_VERSION = 3;

/**
 * The server's project format from document schema 4: the project content and each member's workspace, saved apart
 * (`projectHalves.ts`). Every save this build makes is of that format, and it reads nothing newer.
 */
export const PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION = 4;

/**
 * A workspace half's own schema. 5 holds the project's workspace and views; a half stamped with the split schema (4)
 * holds the layout a project stored before structure and view split, which this build migrates on load and which a
 * build that reads only 4 refuses to read in a 5.
 */
export const PROJECT_WORKSPACE_SCHEMA_VERSION = 5;
