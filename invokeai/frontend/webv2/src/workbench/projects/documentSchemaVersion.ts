/** The whole-document schema the browser works in, and a project the server holds whole is saved in. */
export const PROJECT_DOCUMENT_SCHEMA_VERSION = 3;

/**
 * The server's project format from document schema 4: the project content and each member's workspace, saved apart
 * (`projectHalves.ts`). Every save this build makes is of that format, and it reads nothing newer.
 */
export const PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION = 4;
